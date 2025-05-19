#!/usr/bin/env python
# Purpose of app.py :

#This is the main server file for SageMaker endpoint deployment

#It handles model loading, inference requests, and server management

#It's the interface between SageMaker and your PyTorch model

# 1. Client sends POST to /invocations
# 2. Flask receives request
# 3. Data is parsed from CSV
# 4. PredictionService processes data
# 5. Model makes prediction
# 6. Response is formatted and returned

import json
import io
import sys
import os
import signal
import traceback
import flask
import multiprocessing
import subprocess
import torch
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import model
import boto3

prefix = '/opt/ml/'
model_path = os.path.join(prefix, 'model')
sys.path.insert(0, model_path)  # Add model directory to Python path
model_cache = {}

def load_model():
    """Loads PyTorch model from disk"""
    model_file = os.path.join(model_path, 'model.pt')
    if not os.path.exists(model_file):
        raise Exception(f'Model not found at {model_file}')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.jit.load(model_file, map_location=device)
    model.eval()
    return model

def sigterm_handler(nginx_pid, gunicorn_pid):
    """
    Purpose: Handles graceful shutdown when SageMaker needs to terminate the endpoint
    
    Parameters:
    - nginx_pid: Process ID of the nginx server
    - gunicorn_pid: Process ID of the gunicorn server
    
    Actions:
    1. Sends SIGQUIT signal to nginx (graceful shutdown)
    2. Sends SIGTERM signal to gunicorn (graceful shutdown)
    3. Exits the program
    """
    try:
        os.kill(nginx_pid, signal.SIGQUIT)
    except OSError:
        pass
    try:
        os.kill(gunicorn_pid, signal.SIGTERM)
    except OSError:
        pass
    sys.exit(0)

def start_server():
    """Start the server with nginx and gunicorn"""
    print(f'Starting the inference server with {model_server_workers} workers.')
    print(f'PyTorch Version: {torch.__version__}')

    # Link the log streams to stdout/err
    subprocess.check_call(['ln', '-sf', '/dev/stdout', '/var/log/nginx/access.log'])
    subprocess.check_call(['ln', '-sf', '/dev/stderr', '/var/log/nginx/error.log'])

    nginx = subprocess.Popen(['nginx', '-c', '/opt/program/nginx.conf'])
    gunicorn = subprocess.Popen(['gunicorn',
                               '--timeout', str(model_server_timeout),
                               '-k', 'gevent',
                               '-b', 'unix:/tmp/gunicorn.sock',
                               '-w', str(model_server_workers),
                               'wsgi:app'])

    signal.signal(signal.SIGTERM, lambda a, b: sigterm_handler(nginx.pid, gunicorn.pid))

    # If either subprocess exits, so do we
    pids = set([nginx.pid, gunicorn.pid])
    while True:
        pid, _ = os.wait()
        if pid in pids:
            break

    sigterm_handler(nginx.pid, gunicorn.pid)
    print('Inference server exiting')
    
class PredictionService(object):
    pytorch_model = None                              # holds the PyTorch model
    feature_config = None                             # holds the feature configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    @classmethod
    def get_model(cls):
        """Loads model if not loaded, returns cached model if already loaded"""
        if cls.pytorch_model is None:
            cls.pytorch_model = load_model()
        return cls.pytorch_model
    
    @classmethod
    def _load_feature_config(cls):
        """Load feature configuration from local path or S3"""
        try:
            model_path = '/opt/ml/model'
            feature_config_path = os.path.join(model_path, 'feature_config.json')
    
            # Try local path first
            if os.path.exists(feature_config_path):
                with open(feature_config_path, 'r') as f:
                    feature_config = json.load(f)
                print(f"✅ Loaded feature_config.json from local model path")
                return feature_config
    
            # Try S3 if local file doesn't exist
            print(f"⚠️ feature_config.json not found locally. Attempting to fetch from S3...")
            pipeline_bucket = os.environ.get('PIPELINE_BUCKET')
            if pipeline_bucket:
                s3_key = 'feature_config/feature_config.json'
                try:
                    boto3.client('s3').download_file(
                        pipeline_bucket,
                        s3_key,
                        feature_config_path
                    )
                    with open(feature_config_path, 'r') as f:
                        feature_config = json.load(f)
                    print(f"✅ Successfully loaded feature_config.json from s3://{pipeline_bucket}/{s3_key}")
                    return feature_config
                except Exception as e:
                    print(f"❌ Failed to download feature_config.json from S3: {e}")
            
            print("⚠️ No feature configuration available")
            return None
            
        except Exception as e:
            print(f"Error loading feature configuration: {e}")
            return None

    @classmethod
    def predict(cls, input_data):
        """Handles prediction pipeline: preprocessing → inference → postprocessing"""
        try:
            print("📌 Inside predict()")
            # Get or load the model
            model = cls.get_model() 
            
            # Load feature configuration
            feature_config = cls._load_feature_config()
            
            if feature_config:
                # Split features into categorical and continuous
                cat_features = feature_config.get('categorical', {})
                cont_features = feature_config.get('continuous', {})
                
                # Prepare categorical features
                X_cat = None
                if cat_features:
                    cat_cols = list(cat_features.keys())
                    X_cat_df = input_data[cat_cols]
                    
                    # Handle unknown categories
                    for col in cat_cols:
                        known_categories = set(cat_features[col]['classes'])
                        mask_unknown = ~X_cat_df[col].isin(known_categories)
                        if mask_unknown.any():
                            print(f"⚠️ Found unknown categories in {col}")
                            most_frequent = cat_features[col]['most_frequent']
                            X_cat_df.loc[mask_unknown, col] = most_frequent
                    
                    # Convert categories to indices
                    X_cat = []
                    for col in cat_cols:
                        # Map categories to indices based on classes list
                        cat_to_idx = {cat: idx for idx, cat in enumerate(cat_features[col]['classes'])}
                        X_cat.append([cat_to_idx.get(x, 0) for x in X_cat_df[col]])
                    X_cat = torch.LongTensor(X_cat).T.to(cls.device)
                
                # Prepare continuous features
                X_cont = None
                if cont_features:
                    cont_cols = list(cont_features.keys())
                    X_cont_df = input_data[cont_cols]
                    
                    # Apply scaling
                    for col in cont_cols:
                        mean = cont_features[col]['mean']
                        scale_mean = cont_features[col]['scale_mean']
                        scale_var = cont_features[col]['scale_var']
                        
                        # First impute missing values with mean
                        X_cont_df[col] = X_cont_df[col].fillna(mean)
                        
                        # Then apply standardization
                        X_cont_df[col] = (X_cont_df[col] - scale_mean) / np.sqrt(scale_var)
                    
                    X_cont = torch.FloatTensor(X_cont_df.values).to(cls.device)
                
                print(f"🔹 Input shapes - Categorical: {X_cat.shape if X_cat is not None else 'None'}, "
                      f"Continuous: {X_cont.shape if X_cont is not None else 'None'}") 
                # Make prediction
                try:
                    with torch.no_grad():
                        prediction = model.forward(X_cat, X_cont)
                except Exception as e:
                    print(f"🔥 model.forward() failed: {e}")
                    import traceback
                    traceback.print_exc()
                    raise
            else:
                raise ValueError("No feature configuration available for preprocessing")
            
            return prediction.cpu().numpy()
    
        except Exception as e:
            print(f'Prediction error: {str(e)}')
            raise
        
# The flask app for serving predictions
app = flask.Flask(__name__)

@app.route('/ping', methods=['GET'])
def ping():
    """Determine if the container is working and healthy."""
    try:
        health = PredictionService.get_model() is not None
        status = 200 if health else 404
    except Exception as e:
        print(f"Health check failed: {str(e)}")
        status = 404
    return flask.Response(response='\n', status=status, mimetype='application/json')
    
@app.route('/invocations', methods=['POST'])
def invoke():
    """Main prediction endpoint for SageMaker"""
    if flask.request.content_type != 'text/csv':
        return flask.Response(
            response='This predictor only supports CSV data',
            status=415, mimetype='text/plain'
        )

    try:
        # Read input data
        data = flask.request.data.decode('utf-8')
        input_data = pd.read_csv(io.StringIO(data), header=0)
        print(f"Received request with columns: {input_data.columns.tolist()}")

        # Make predictions
        predictions = PredictionService.predict(input_data)

        # Return predictions as CSV
        out = io.StringIO()
        pd.DataFrame({'results': predictions.flatten()}).to_csv(out, header=False, index=False)
        result = out.getvalue()

        return flask.Response(response=result, status=200, mimetype='text/csv')

    except Exception as e:
        error_message = f'Prediction failed: {str(e)}\n{traceback.format_exc()}'
        print(error_message)
        return flask.Response(
            response=error_message,
            status=500, mimetype='text/plain'
        )

        
if __name__ == '__main__':
    print(f"PyTorch Version: {torch.__version__}")
    if len(sys.argv) < 2 or (not sys.argv[1] in ["serve", "train", "test"]):
        raise Exception("Invalid argument: you must specify 'train' for training mode, 'serve' for predicting mode or 'test' for local testing.")

    train = sys.argv[1] == "train"
    test = sys.argv[1] == "test"

    if train:
        model.train()
        
    elif test:
        algo = 'PyTorchRegression'
        if model_cache.get(algo) is None:
            model_cache[algo] = load_model()
        req = eval(sys.argv[2])
        # Convert input to tensor
        input_tensor = torch.FloatTensor(req).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        # Make prediction
        with torch.no_grad():
            prediction = model_cache[algo](input_tensor)
        print(prediction.cpu().numpy())

    else:
        model_server_timeout = int(os.environ.get('MODEL_SERVER_TIMEOUT', 60))
        model_server_workers = int(os.environ.get('MODEL_SERVER_WORKERS', multiprocessing.cpu_count()))
        start_server()
