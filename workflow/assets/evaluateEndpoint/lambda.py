import os
import io
import json
import logging
import boto3
import time
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from botocore.exceptions import ClientError

# Setup logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize AWS clients
s3 = boto3.client("s3")
sm_client = boto3.client("sagemaker-runtime")

def evaluate_model(bucket, key, endpoint_name, target_column):
    """
    Evaluates PyTorch model predictions on the testing dataset.
    
    Args:
        bucket (str): S3 bucket containing test data
        key (str): S3 key for test data
        endpoint_name (str): SageMaker endpoint name
    
    Returns:
        tuple: Ground truth labels, predictions, and response times
    """
    try:
        # Load test data
        logger.info(f"Loading test data from s3://{bucket}/{key}")
        obj = s3.get_object(Bucket=bucket, Key=key)
        test_df = pd.read_csv(io.BytesIO(obj['Body'].read()))
        
        # Separate features and target
        feature_columns = [col for col in test_df.columns if col != target_column]
        X_test = test_df[feature_columns]
        y_test = test_df[target_column].values
        
        # Initialize lists for storing results
        predictions = []
        response_times = []
        
        # Process each row
        logger.info(f"Starting predictions for {len(X_test)} samples")
        
        for idx, row in X_test.iterrows():
            buf = io.StringIO()
            # Prepare payload
            row.to_frame().T.to_csv(buf, index=False, header=True)
            payload = buf.getvalue()
            # Time the prediction
            start_time = time.time()
            try:
                response = sm_client.invoke_endpoint(
                    EndpointName=endpoint_name,
                    ContentType="text/csv",
                    Body=payload
                )
                response_time = time.time() - start_time
                response_times.append(response_time)
                
                # Parse response
                result = float(response['Body'].read().decode('utf-8').strip())
                predictions.append(result)
                
            except ClientError as e:
                logger.error(f"Error invoking endpoint: {str(e)}")
                raise
                
            # Log progress periodically
            if (idx + 1) % 100 == 0:
                logger.info(f"Processed {idx + 1} samples")
        
        return y_test, predictions, response_times
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise

def calculate_metrics(y_true, y_pred):
    """
    Calculate regression metrics for model evaluation.
    """
    return {
        "mse": float(mean_squared_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred, squared=False)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred))
    }

def handler(event, context):
    """
    Lambda handler for model evaluation.
    """
    logger.info("Starting model evaluation")
    logger.debug(f"Event: {json.dumps(event)}")
    
    try:
        # Validate input parameters
        required_params = ["Bucket", "Key", "Endpoint_Name", "Output_Key","Target_Column"]
        for param in required_params:
            if param not in event:
                raise KeyError(f"Missing required parameter: {param}")
        
        # Extract parameters
        bucket = event["Bucket"]
        key = event["Key"]
        endpoint_name = event["Endpoint_Name"]
        output_key = event["Output_Key"]
        target_column = event["Target_Column"]
        # Evaluate model
        y_test, predictions, response_times = evaluate_model(bucket, key, endpoint_name, target_column)
        
        # Calculate metrics
        metrics = calculate_metrics(y_test, predictions)
        
        # Add response time statistics
        metrics.update({
            "avg_response_time": float(np.mean(response_times)),
            "p95_response_time": float(np.percentile(response_times, 95)),
            "max_response_time": float(np.max(response_times))
        })
        
        # Save metrics to S3
        evaluation_report = {
            "regression_metrics": metrics,
            "timestamp": time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())
        }
        
        try:
            s3.put_object(
                Bucket=bucket,
                Key=f"{output_key}/evaluation.json",
                Body=json.dumps(evaluation_report, indent=4)
            )
        except ClientError as e:
            logger.error(f"Error saving evaluation results: {str(e)}")
            raise
        
        logger.info(f"Evaluation completed. RMSE: {metrics['rmse']:.4f}")
        
        return {
            "statusCode": 200,
            "Result": metrics["rmse"],
            "Metrics": metrics
        }
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        return {
            "statusCode": 500,
            "error": str(e)
        }
