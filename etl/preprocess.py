import os
import sys
import boto3
import numpy as np
import pandas as pd
import sklearn
from awsglue.utils import getResolvedOptions
from io import StringIO
import json
import logging
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout  # Force output to stdout
)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self, s3_output_bucket=None, s3_output_key_prefix=None):
        """Initialize the DataPreprocessor with configuration"""
        self.cont_cols = ["length","diameter","height","whole_weight","shucked_weight","viscera_weight","shell_weight"]
        self.cat_cols = ["sex"]
        self.target_column = "rings"
        self.feature_config = {}
        self.label_encoders = {}
        self.imputers = {}
        self.scalers = {}
        self.s3_output_bucket = s3_output_bucket
        self.s3_output_key_prefix = s3_output_key_prefix
        self.s3_client = boto3.Session().resource('s3')

    def split_data(self, df, train_percent=0.8, validate_percent=0.19, seed=42):
        """Split dataset into train, validate, test, and baseline sets"""
        logger.info("Splitting data into train, validate, test, and baseline sets...")
        np.random.seed(seed)
        perm = np.random.permutation(df.index)
        m = len(df.index)
        train_end = int(train_percent * m)
        validate_end = int(validate_percent * m) + train_end
        
        train = df.iloc[perm[:train_end]].copy()
        validate = df.iloc[perm[train_end:validate_end]].copy()
        test = df.iloc[perm[validate_end:]].copy()
        
        return [('train', train), ('test', test), ('validate', validate), ('baseline', train.copy())]

    def fit_transformers(self, data):
        """Fit all transformers on complete dataset.
        This method is responsible for preparing the data transformers that 
        will be used to process categorical and continuous features."""
        logger.info("Fitting transformers on complete dataset...")
        
        try:
            # Fit categorical transformers
            for col in self.cat_cols:
                logger.info(f"Processing categorical column: {col}")
                # Fit imputer
                imputer = SimpleImputer(strategy='most_frequent')
                imputer.fit(data[[col]])
                self.imputers[col] = imputer
                
                # Fit label encoder
                le = LabelEncoder()
                imputed_values = imputer.transform(data[[col]]).ravel() #flatten the 2D array returned by the imputer into a 1D array, which is required by the LabelEncoder.
                le.fit(imputed_values)
                self.label_encoders[col] = le
                
                # Store mapping information
                self.feature_config.setdefault('categorical', {})[col] = {
                    'classes': le.classes_.tolist(),
                    'most_frequent': imputed_values[0]
                }
            
            # Fit continuous transformers
            self.feature_config['continuous'] = {}
            for col in self.cont_cols:
                logger.info(f"Processing continuous column: {col}")
                # Fit imputer
                imputer = SimpleImputer(strategy='mean')
                imputer.fit(data[[col]])
                self.imputers[col] = imputer
                
                # Get imputed values and fit scaler
                imputed_values = imputer.transform(data[[col]])
                scaler = StandardScaler()
                scaler.fit(imputed_values)
                self.scalers[col] = scaler   
                
                # Store continuous transformation parameters
                self.feature_config['continuous'][col] = {
                    'mean': imputer.statistics_.tolist()[0],
                    'scale_mean': scaler.mean_.tolist()[0],
                    'scale_var': scaler.var_.tolist()[0]
                }
            
            # Store all feature names in order
            self.feature_config['feature_names'] = self.cont_cols + self.cat_cols
            
        except Exception as e:
            logger.error(f"Error fitting transformers: {str(e)}")
            logger.error(f"Data info:")
            logger.error(data.info())
            raise

    def transform_features(self, data, is_training=False):
        """Transform features using fitted transformers"""
        try:
            # Create DataFrame with same index as input data
            transformed_data = pd.DataFrame(index=data.index)
            
            # Transform continuous features
            for col in self.cont_cols:
                logger.info(f"Transforming continuous column: {col}")
                
                cont_data = data[[col]]
                
                # Apply imputation
                imputed_data = self.imputers[col].transform(cont_data)
                
                # Apply scaling only to training datasets (train, validation, baseline)
                scaled_data = self.scalers[col].transform(imputed_data)
                transformed_data[col] = scaled_data.ravel()
            
            # Transform categorical features
            for col in self.cat_cols:
                logger.info(f"Transforming categorical column: {col}")
                imputed_values = self.imputers[col].transform(data[[col]]).ravel()
                transformed_data[col] = self.label_encoders[col].transform(imputed_values)
            
            # Ensure correct column order
            transformed_data = transformed_data[self.feature_config['feature_names']]
            return transformed_data
            
        except Exception as e:
            logger.error(f"Error in feature transformation: {str(e)}")
            raise

    def preprocess_data(self, data):
        """Main preprocessing function"""
        try:
            logger.info("Starting preprocessing pipeline...")
            
            # Check for missing values in target column
            missing_target = data[self.target_column].isnull().sum()
            if missing_target > 0:
                logger.warning(f"Found {missing_target} missing values in target column {self.target_column}") 
            
            # Reorder columns
            ordered_columns = [self.target_column] + self.cat_cols + self.cont_cols
            data = data[ordered_columns]
            
            # Fit transformers on complete dataset (excluding target)
            logger.info("Fitting transformers on complete dataset...")
            self.fit_transformers(data.drop(columns=[self.target_column]))
            
            # Split data
            datasets = self.split_data(data)
            processed_datasets = {}
            
            # Process each split
            for name, df in datasets:
                logger.info(f"Processing {name} dataset...")
                X = df.drop(columns=[self.target_column])
                y = df[self.target_column]
                
                # Ensure X and y have the same index
                if not X.index.equals(y.index):
                    logger.warning(f"Index mismatch in {name} dataset")
                    common_indices = X.index.intersection(y.index)
                    X = X.loc[common_indices]
                    y = y.loc[common_indices]
                
                if name in ['train', 'validate', 'baseline']:
                    # Transform only training and validation data
                    logger.info(f"Transforming {name} dataset...")
                    X_transformed = self.transform_features(
                        X, 
                        is_training=(name in ['train', 'baseline','validate'])
                    )
                    processed_datasets[name] = (X_transformed, y)
                else:
                    # Store test data without transformation
                    logger.info(f"Storing {name} dataset without transformation...")
                    processed_datasets[name] = (X, y)
            
            return processed_datasets
                
        except Exception as e:
            logger.error(f"Error in data preprocessing: {str(e)}")
            raise


    def save_to_s3(self, processed_data):
        """Save processed data and feature configuration to S3"""
        try:
            if not self.s3_output_bucket or not self.s3_output_key_prefix:
                raise ValueError("S3 bucket and prefix must be set")
    
            # Initialize S3 client once
            s3_client = boto3.client('s3')
            
            logger.info("Saving feature configuration to S3...")
            
            # Save feature config directly to S3 
            feature_config_key = "feature_config/feature_config.json"
            
            logger.info(f"Uploading feature config to S3: s3://{self.s3_output_bucket}/{feature_config_key}")
            
            s3_client.put_object(
                Bucket=self.s3_output_bucket,
                Key=feature_config_key,
                Body=json.dumps(self.feature_config, indent=2)
            )
            
            logger.info(f"Successfully uploaded feature config to S3: {feature_config_key}")
    
            # Save datasets
            for name, (X, y) in processed_data.items():
                logger.info(f"Processing {name} dataset...")
                
                # Check if X and y have the same length
                if len(X) != len(y):
                    logger.warning(f"Length mismatch in {name} dataset: X={len(X)}, y={len(y)}")
                    
                    # Ensure X and y have the same indices
                    #If both X and y have indices (i.e., they're pandas DataFrames or Series), 
                    # it finds the common indices using X.index.intersection(y.index) and filters both 
                    # X and y to only include those common indices. If either X or y doesn't have indices 
                    # (e.g., they're numpy arrays), it truncates both to the shorter length.
                    if hasattr(X, 'index') and hasattr(y, 'index'):
                        common_indices = X.index.intersection(y.index)
                        logger.info(f"Using {len(common_indices)} common indices")
                        X = X.loc[common_indices]
                        y = y.loc[common_indices]
                    else:
                        # If X or y don't have indices (e.g., numpy arrays), use the shorter length
                        min_len = min(len(X), len(y))
                        logger.info(f"Using first {min_len} rows")
                        X = X[:min_len]
                        y = y[:min_len]
                
                if name == 'test':
                    # For test data, use original feature names
                    output_data = pd.DataFrame(
                        X,
                        columns=self.cat_cols + self.cont_cols
                    )
                else:
                    # For transformed data, use feature names from config
                    output_data = pd.DataFrame(
                        X, 
                        columns=self.feature_config['feature_names']
                    )
                
                # Insert target column and check for NaN values
                output_data.insert(0, self.target_column, y)
                
                # Check for NaN values in the target column
                nan_count = output_data[self.target_column].isna().sum()
                if nan_count > 0:
                    logger.warning(f"Found {nan_count} NaN values in {self.target_column} column in {name} dataset")
                    
                    # Option: Drop rows with NaN target values
                    output_data = output_data.dropna(subset=[self.target_column])
                    logger.info(f"After dropping rows with NaN target values, {name} dataset shape: {output_data.shape}")
                
                # Convert to CSV
                csv_buffer = StringIO()
                output_data.to_csv(csv_buffer, index=False)
                
                # Determine S3 prefix
                if name == 'test':
                    s3_prefix = 'testing'
                elif name == 'baseline':
                    s3_prefix = 'baseline'
                else:
                    s3_prefix = 'training'
    
                # Upload to S3
                s3_key = f"{self.s3_output_key_prefix}/{s3_prefix}/{name}.csv"
                
                logger.info(f"Uploading {name} dataset to S3: s3://{self.s3_output_bucket}/{s3_key}")
                logger.info(f"Dataset shape: {output_data.shape}")
                
                s3_client.put_object(
                    Bucket=self.s3_output_bucket,
                    Key=s3_key,
                    Body=csv_buffer.getvalue()
                )
                
                logger.info(f"Successfully uploaded {name} dataset to S3: {s3_key}")
    
            logger.info("Completed all S3 uploads successfully")
            
        except Exception as e:
            logger.error(f"Error saving to S3: {str(e)}")
            logger.error(f"Error details: {str(e.__class__.__name__)}")
            raise



def main():
    """Main execution function"""
    try:
        # Get arguments from Glue job
        args = getResolvedOptions(sys.argv, 
                                ['S3_INPUT_BUCKET', 
                                 'S3_INPUT_KEY_PREFIX', 
                                 'S3_OUTPUT_BUCKET', 
                                 'S3_OUTPUT_KEY_PREFIX'])

        logger.info("Starting data preprocessing job...")
        
        dtype_dict = {
            'sex': str,
            'length': float,
            'diameter': float,
            'height': float,
            'whole_weight': float,
            'shucked_weight': float,
            'viscera_weight': float,
            'shell_weight': float,
            'rings': float
        }

        # Download data from S3
        logger.info("Downloading input data from S3...")
        client = boto3.client('s3')
        csv_obj = client.get_object(
            Bucket=args['S3_INPUT_BUCKET'],
            Key=os.path.join(args['S3_INPUT_KEY_PREFIX'], 'abalone.csv')
        ) 
        body = csv_obj['Body']
        csv_string = body.read().decode('utf-8')
        data = pd.read_csv(StringIO(csv_string), dtype=dtype_dict)
        logger.info(f"Loaded data shape: {data.shape}")
        
        # Check for missing values in raw data
        missing_values = data.isnull().sum()
        logger.info("Missing values in raw data:")
        for col, count in missing_values.items():
            if count > 0:
                logger.info(f"{col}: {count}")
        
        # Initialize preprocessor
        preprocessor = DataPreprocessor(
            s3_output_bucket=args['S3_OUTPUT_BUCKET'],
            s3_output_key_prefix=args['S3_OUTPUT_KEY_PREFIX']
        )

        # Process data
        processed_data = preprocessor.preprocess_data(data)
        
        # Check for missing values in processed data
        for name, (X, y) in processed_data.items():
            nan_count = pd.isna(y).sum()
            if nan_count > 0:
                logger.warning(f"Found {nan_count} NaN values in target column of {name} dataset")
                logger.warning(f"This is {nan_count/len(y)*100:.2f}% of the dataset")

        # Save to S3
        preprocessor.save_to_s3(processed_data)

        logger.info("Data preprocessing job completed successfully")

    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise


if __name__ == "__main__":
    main()
