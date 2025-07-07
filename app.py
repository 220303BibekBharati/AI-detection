from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import os
from collections import Counter
from sklearn.utils import resample
import shap
import numpy as np

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model_iforest = None
model_mlp = None
X_columns = None
selected_model_type = None
mlp_scaler = None

@app.route('/upload', methods=['POST'])
def upload_csv():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    print(f"Saving file to: {os.path.abspath(filepath)}")  # Debug: log the save path
    file.save(filepath)
    return jsonify({'message': 'File uploaded successfully', 'filename': file.filename})

@app.route('/train', methods=['POST'])
def train_model():
    data = request.get_json()
    filename = data.get('filename')
    model_type = data.get('model_type', 'Isolation Forest')
    if not filename:
        return jsonify({'error': 'No filename provided'}), 400
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404
    df = pd.read_csv(filepath)
    # Use all numeric columns except 'label' for training
    features = df.select_dtypes(include='number').copy()
    if 'label' in features:
        features = features.drop('label', axis=1)
    # Fill missing values with column mean
    features = features.fillna(features.mean())
    global model_iforest, model_mlp, X_columns, selected_model_type
    X_columns = features.columns.tolist()
    metrics = {}
    if model_type == 'Isolation Forest':
        model_iforest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
        model_iforest.fit(features)
        selected_model_type = 'Isolation Forest'
        # If labels are present, calculate metrics
        if 'label' in df:
            preds = model_iforest.predict(features)
            preds = (preds == -1).astype(int)
            y_true = df['label'].values
            metrics = {
                'accuracy': accuracy_score(y_true, preds),
                'precision': precision_score(y_true, preds, zero_division=0),
                'recall': recall_score(y_true, preds, zero_division=0),
                'f1': f1_score(y_true, preds, zero_division=0)
            }
    elif model_type == 'MLP Model':
        if 'label' not in df:
            return jsonify({'error': 'No label column for supervised training'}), 400
        # Robust preprocessing: scale features
        scaler = StandardScaler()
        X = features.values
        y = df['label'].values
        X_scaled = scaler.fit_transform(X)
        # Train/test split for real-world evaluation
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
        # Log class distribution
        class_dist = dict(Counter(y_train))
        # Oversample the minority class in the training set
        X_train_df = pd.DataFrame(X_train)
        y_train_df = pd.Series(y_train)
        Xy_train = X_train_df.copy()
        Xy_train['label'] = y_train_df.values
        # Separate majority and minority
        majority = Xy_train[Xy_train['label'] == 0]
        minority = Xy_train[Xy_train['label'] == 1]
        if len(minority) > 0 and len(majority) > 0:
            minority_upsampled = resample(minority, replace=True, n_samples=len(majority), random_state=42)
            upsampled = pd.concat([majority, minority_upsampled])
            X_train_bal = upsampled.drop('label', axis=1).values
            y_train_bal = upsampled['label'].values
        else:
            X_train_bal = X_train
            y_train_bal = y_train
        model_mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, alpha=0.001, random_state=42, early_stopping=True, n_iter_no_change=10)
        model_mlp.fit(X_train_bal, y_train_bal)
        selected_model_type = 'MLP Model'
        # Evaluate on test set
        preds = model_mlp.predict(X_test)
        cm = confusion_matrix(y_test, preds)
        recall_val = recall_score(y_test, preds, zero_division=0)
        metrics = {
            'accuracy': accuracy_score(y_test, preds),
            'precision': precision_score(y_test, preds, zero_division=0),
            'recall': recall_val,
            'f1': f1_score(y_test, preds, zero_division=0),
            'confusion_matrix': cm.tolist(),
            'class_distribution': {str(k): int(v) for k, v in class_dist.items()},
            'train_class_distribution': {str(k): int(v) for k, v in Counter(y_train_bal).items()}
        }
        if recall_val == 0:
            metrics['warning'] = 'Model did not detect any anomalies in the test set. Consider more feature engineering.'
        # Save scaler for use in prediction
        global mlp_scaler
        mlp_scaler = scaler
    else:
        return jsonify({'error': 'Unknown model type'}), 400
    return jsonify({'message': 'Model trained', 'features': X_columns, 'model_type': selected_model_type, 'metrics': metrics})

@app.route('/predict', methods=['POST'])
def predict_anomalies():
    data = request.get_json()
    filename = data.get('filename')
    model_type = data.get('model_type', 'Isolation Forest')
    if not filename:
        return jsonify({'error': 'No filename provided'}), 400
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404
    df = pd.read_csv(filepath)
    features = df.select_dtypes(include='number').copy()
    if 'label' in features:
        features = features.drop('label', axis=1)
    features = features.fillna(features.mean())
    global model_iforest, model_mlp, X_columns, selected_model_type
    anomalies = []
    anomaly_indices = []
    anomaly_amounts = []
    all_amounts = df['amount'].tolist() if 'amount' in df else []
    if model_type == 'Isolation Forest' and model_iforest is not None:
        preds = model_iforest.predict(features)
        df['iso_pred'] = (preds == -1).astype(int)
        for idx, row in df.iterrows():
            if row['iso_pred'] == 1:
                anomalies.append({
                    'log_id': row.get('log_id', '-'),
                    'member_id': row.get('member_id', '-'),
                    'transaction_type': row.get('transaction_type', '-'),
                    'amount': row.get('amount', '-'),
                    'anomaly': 1
                })
                anomaly_indices.append(idx)
                anomaly_amounts.append(row.get('amount', 0))
        chart_data = {
            'amounts': all_amounts,
            'anomaly_indices': anomaly_indices,
            'anomaly_amounts': anomaly_amounts
        }
    elif model_type == 'MLP Model' and model_mlp is not None:
        # Use the same scaler as during training
        global mlp_scaler
        X = features.values
        X_scaled = mlp_scaler.transform(X)
        preds = model_mlp.predict(X_scaled)
        df['mlp_pred'] = preds
        for idx, row in df.iterrows():
            if row['mlp_pred'] == 1:
                anomalies.append({
                    'log_id': row.get('log_id', '-'),
                    'member_id': row.get('member_id', '-'),
                    'transaction_type': row.get('transaction_type', '-'),
                    'amount': row.get('amount', '-'),
                    'anomaly': 1
                })
                anomaly_indices.append(idx)
                anomaly_amounts.append(row.get('amount', 0))
        chart_data = {
            'amounts': all_amounts,
            'anomaly_indices': anomaly_indices,
            'anomaly_amounts': anomaly_amounts
        }
    else:
        return jsonify({'error': 'Model not trained or unknown model type'}), 400
    return jsonify({'anomalies': anomalies, 'chart_data': chart_data})

@app.route('/data', methods=['GET'])
def get_csv_data():
    filename = request.args.get('filename')
    if not filename:
        return jsonify({'error': 'No filename provided'}), 400
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404
    try:
        df = pd.read_csv(filepath)
        data = df.to_dict(orient='records')
        return jsonify({'data': data})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/shap', methods=['POST'])
def shap_explain():
    data = request.get_json()
    filename = data.get('filename')
    model_type = data.get('model_type', 'Isolation Forest')
    index = data.get('index', None)
    if not filename:
        return jsonify({'error': 'No filename provided'}), 400
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404
    df = pd.read_csv(filepath)
    features = df.select_dtypes(include='number').copy()
    if 'label' in features:
        features = features.drop('label', axis=1)
    features = features.fillna(features.mean())
    global model_iforest, model_mlp, mlp_scaler
    try:
        if model_type == 'Isolation Forest' and model_iforest is not None:
            explainer = shap.TreeExplainer(model_iforest)
            shap_values = explainer.shap_values(features)
            global_importance = np.abs(shap_values).mean(axis=0)
            result = {
                'feature_names': list(features.columns),
                'global_importance': global_importance.tolist(),
            }
            if index is not None and 0 <= index < len(features):
                result['local_shap'] = shap_values[index].tolist()
        elif model_type == 'MLP Model' and model_mlp is not None:
            # Use the same scaler as during training
            X_scaled = mlp_scaler.transform(features.values)
            # Use a small background sample for KernelExplainer
            background = shap.utils.sample(X_scaled, min(100, X_scaled.shape[0]), random_state=42)
            explainer = shap.KernelExplainer(model_mlp.predict_proba, background)
            # For speed, only explain a sample of points
            sample_idx = np.arange(X_scaled.shape[0])
            if X_scaled.shape[0] > 200:
                sample_idx = np.random.choice(X_scaled.shape[0], 200, replace=False)
            shap_values = explainer.shap_values(X_scaled[sample_idx], nsamples=100)
            global_importance = np.abs(shap_values[1]).mean(axis=0)  # class 1 (anomaly)
            result = {
                'feature_names': list(features.columns),
                'global_importance': global_importance.tolist(),
                'explained_indices': sample_idx.tolist()
            }
            if index is not None and 0 <= index < X_scaled.shape[0]:
                local_shap = explainer.shap_values(X_scaled[index:index+1], nsamples=100)
                result['local_shap'] = local_shap[1][0].tolist()
        else:
            return jsonify({'error': 'Model not trained or unknown model type'}), 400
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': f'SHAP explanation failed: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)