import os
import sys
import json
import re
import traceback
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn import preprocessing
from torch.utils.data import Dataset, DataLoader
import logging
import shutil
import boto3

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Declare communication channel between Sagemaker and container
prefix = '/opt/ml'
input_path = os.path.join(prefix, 'input/data')
output_path = os.path.join(prefix, 'output')
model_path = os.path.join(prefix, 'model')
param_path = os.path.join(prefix, 'input/config/hyperparameters.json') 

class RegressionDataset(Dataset):
    def __init__(self, cats, conts, target):
        """
        Initialize dataset with categorical, continuous features and target
        """
        self.cats = torch.tensor(cats.values, dtype=torch.int64) if cats is not None else None
        self.conts = torch.tensor(conts.values, dtype=torch.float32) if conts is not None else None
        self.target = torch.tensor(target.values, dtype=torch.float32)

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        cat_data = self.cats[idx] if self.cats is not None else torch.tensor([])
        cont_data = self.conts[idx] if self.conts is not None else torch.tensor([])
        return cat_data, cont_data, self.target[idx]

def ifnone(a, b):
    """Return `a` if `a` is not None, otherwise return `b`"""
    return b if a is None else a

def listify(p=None, q=None):
    """Make `p` listy and the same length as `q`"""
    if p is None:
        p = []
    elif not isinstance(p, (list, tuple)):
        p = [p]
    
    n = q if isinstance(q, int) else len(q) if q is not None else 1
    if len(p) == 1:
        p = p * n
    return list(p)

def bn_drop_lin(n_in:int, n_out:int, bn:bool=True, p:float=0., actn=None):
    """
    Sequence of batchnorm, dropout, linear layers with optional activation.
    """
    layers = [nn.BatchNorm1d(n_in)] if bn else []
    if p != 0:
        layers.append(nn.Dropout(p))
    layers.append(nn.Linear(n_in, n_out))
    if actn is not None:
        layers.append(actn)
    return layers

class TabularModel(nn.Module):
    "Basic model for tabular data."
    def __init__(self, emb_szs, n_cont:int, out_sz:int, layers, ps=None,
                 emb_drop:float=0., y_range=None, use_bn:bool=True, bn_final:bool=False):
        """
        Initialize the model
        
        Args:
            emb_szs (list): List of tuples (vocab_size, emb_dim) for categorical variables
            n_cont (int): Number of continuous variables
            out_sz (int): Output size
            layers (list): List of hidden layer sizes
            ps (list): List of dropout probabilities
            emb_drop (float): Embedding dropout
            y_range (tuple): Range for output variable
            use_bn (bool): Whether to use batch normalization
            bn_final (bool): Whether to use batch normalization in final layer
        """
        super().__init__()
        
        # Make drop out probabilities available as per layers - List operations
        ps = ifnone(ps, [0]*len(layers))
        ps = listify(ps, layers)
        
        # Initialize embeddings with better weight initialization
        self.embeds = nn.ModuleList([nn.Embedding(ni, nf) for ni,nf in emb_szs])
        for emb in self.embeds:
            nn.init.xavier_uniform_(emb.weight)
            
        self.emb_drop = nn.Dropout(emb_drop) #type: torch.nn.modules.dropout.Dropout
        self.bn_cont = nn.BatchNorm1d(n_cont) if n_cont > 0 else None #type torch.nn.modules.batchnorm.
        
        # Calculate input size
        n_emb = sum(e.embedding_dim for e in self.embeds) # n_emb = 17 , type: int
        self.n_emb,self.n_cont,self.y_range = n_emb,n_cont,y_range
        
        # Layer architecture
        sizes = [n_emb + n_cont] + layers + [out_sz] #typeL list, len: 4
        actns = [nn.ReLU(inplace=True) for _ in range(len(sizes)-2)] + [None] #type: list, len: 3.  the last in None because we finish with linear
        # Create layers with improved initialization
        layers = []
        for i,(n_in,n_out,dp,act) in enumerate(zip(sizes[:-1],sizes[1:],[0.]+ps,actns)):
            layers += bn_drop_lin(n_in, n_out, bn=use_bn and i!=0, p=dp, actn=act)
            # Initialize weights for linear layers
            if isinstance(layers[-1], nn.Linear):
                nn.init.kaiming_normal_(layers[-1].weight)
                if layers[-1].bias is not None:
                    nn.init.constant_(layers[-1].bias, 0)
                    
        if bn_final: 
            layers.append(nn.BatchNorm1d(sizes[-1]))
            
        self.layers = nn.Sequential(*layers)  
        
        # Add model architecture logging
        logger.info(f"Model architecture:")
        logger.info(f"Embedding sizes: {emb_szs}")
        logger.info(f"Continuous variables: {n_cont}")
        logger.info(f"Layer sizes: {sizes}")
        logger.info(f"Dropout rates: {ps}")
        logger.info(f"Batch norm: {use_bn}, Final batch norm: {bn_final}")
        
    def forward(self, x_cat, x_cont):
        """
        Forward pass
        
        Args:
            x_cat (torch.Tensor): Categorical variables
            x_cont (torch.Tensor): Continuous variables
            
        Returns:
            torch.Tensor: Model output
        """
        # Handle categorical features
        if self.n_emb != 0 and x_cat is not None:
            # Process categorical variables
            embeddings = [e(x_cat[:,i]) for i,e in enumerate(self.embeds)]
            x_cat_processed = torch.cat(embeddings, 1)
            x_cat_processed = self.emb_drop(x_cat_processed)
        else:
            x_cat_processed = None
        
        # Handle continuous features
        if self.n_cont != 0 and x_cont is not None:
            x_cont_processed = self.bn_cont(x_cont)
        else:
            x_cont_processed = None
        
        # Combine features
        if x_cat_processed is not None and x_cont_processed is not None:
            # Both categorical and continuous features exist
            x = torch.cat([x_cat_processed, x_cont_processed], 1)
        elif x_cat_processed is not None:
            # Only categorical features exist
            x = x_cat_processed
        elif x_cont_processed is not None:
            # Only continuous features exist
            x = x_cont_processed
        else:
            # No features provided 
            raise ValueError("Neither categorical nor continuous features were provided")
        
        # Forward through layers
        x = self.layers(x)
        
        # Apply output range if specified
        if self.y_range is not None:
            x = (self.y_range[1]-self.y_range[0]) * torch.sigmoid(x) + self.y_range[0]
        
        return x.squeeze()

        
def load_data(file_path):
    """
    Load and preprocess data from CSV file
    
    Args:
        file_path (str): Path to CSV file
        
    Returns:
        pd.DataFrame: Loaded data
    """
    try:
        data = pd.read_csv(file_path)
        logger.info(f"Loaded data from {file_path} with shape: {data.shape}")
        return data
    except Exception as e:
        error_msg = f"Error loading data from {file_path}: {str(e)}"
        logger.error(error_msg) 
        raise
    
def load_training_data():
    """
    Load training data, validation data, and feature configuration
    
    Returns:
        tuple: (train_data, val_data, feature_config)
    """
    try:
        # Set up paths
        channel_name = 'training'
        training_path = os.path.join(input_path, channel_name)
        bucket = os.environ.get('PIPELINE_BUCKET')
        s3_key = 'feature_config/feature_config.json'
        feature_config_local_path = os.path.join(model_path, 'feature_config.json')

        # Download feature config from S3
        if bucket:
            logger.info(f"Attempting to download feature_config.json from s3://{bucket}/{s3_key}")
            try:
                boto3.client('s3').download_file(bucket, s3_key, feature_config_local_path)
                logger.info("✅ feature_config.json downloaded successfully.")
                with open(feature_config_local_path, 'r') as f:
                    feature_config = json.load(f)
            except boto3.exceptions.S3UploadFailedError as e:
                logger.warning(f"⚠️ feature_config.json not found in S3: {e}. Continuing without it.")
                feature_config = {}
            except Exception as e:
                logger.warning(f"⚠️ Could not download feature_config.json: {e}. Proceeding with training anyway.")
                feature_config = {}
        else:
            logger.warning("⚠️ PIPELINE_BUCKET not set. Skipping download of feature_config.json.")
            feature_config = {}

        # Verify input files
        input_files = [os.path.join(training_path, file) for file in os.listdir(training_path)]
        if len(input_files) == 0:
            raise ValueError(f'No files found in {training_path}')
        logger.info(f"Found input files: {input_files}")

        # Load datasets
        train_data = load_data(os.path.join(training_path, 'train.csv'))
        val_data = load_data(os.path.join(training_path, 'validate.csv'))
        
        return train_data, val_data, feature_config

    except Exception as e:
        error_msg = f"Error in load_training_data: {str(e)}"
        logger.error(error_msg) 
        raise
    
def load_hyperparameters():
    """
    Load hyperparameters from SageMaker config
    
    Returns:
        dict: Hyperparameters
    """
    try:
        # Read hyperparameters
        with open(param_path, 'r') as tc:
            params = {}
            for key, value in json.load(tc).items():
                # Convert string values to appropriate types
                if value.lower() == 'true':
                    params[key] = True
                elif value.lower() == 'false':
                    params[key] = False
                elif re.match(r'^\d+\.\d+$', value):
                    params[key] = float(value)
                elif re.match(r'^\d+$', value):
                    params[key] = int(value)
                else:
                    params[key] = value 
        
        logger.info(f"Loaded hyperparameters: {params}")
        return params
    
    except Exception as e:
        error_msg = f"Error loading hyperparameters: {str(e)}"
        logger.error(error_msg) 
        raise

        
def train_epoch(model, train_loader, criterion, optimizer, device, params):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    for batch_idx, (cat, cont, target) in enumerate(train_loader):
        # Move data to device
        if cat.nelement() > 0:
            cat = cat.to(device)
        if cont.nelement() > 0:
            cont = cont.to(device)
        target = target.to(device)

        # Forward pass
        optimizer.zero_grad()
        output = model(cat, cont)
        loss = criterion(output, target)

        # Backward pass with gradient clipping
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), params['gradient_clip_val'])
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 100 == 0:
            logger.info(f'Batch [{batch_idx}/{len(train_loader)}]: Loss: {loss.item():.6f}')

    return total_loss / len(train_loader)

def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for cat, cont, target in val_loader:
            if cat.nelement() > 0:
                cat = cat.to(device)
            if cont.nelement() > 0:
                cont = cont.to(device)
            target = target.to(device)

            output = model(cat, cont)
            loss = criterion(output, target)
            total_loss += loss.item()

    return total_loss / len(val_loader)

def save_model(model, metrics):
    """Save model, feature config, and metrics"""
    try:
        # Save model using torch.jit
        scripted_model = torch.jit.script(model)
        torch.jit.save(scripted_model, os.path.join(model_path, 'model.pt'))
        logger.info("✓ Model saved successfully")
        
        # # Save feature configuration
        # with open(os.path.join(model_path, 'feature_config.json'), 'w') as f:
        #     json.dump(feature_config, f)
        # logger.info("✓ Feature configuration saved successfully")
        
        # Save metrics
        with open(os.path.join(model_path, 'metrics.json'), 'w') as f:
            json.dump(metrics, f)
        logger.info("✓ Training metrics saved successfully")
        
    except Exception as e:
        logger.error(f"Error saving model artifacts: {str(e)}")
        raise

def initialize_model(params, cont_cols, emb_szs, y_range, device):
    """
    Initialize TabularModel with given parameters
    
    Args:
        params (dict): Hyperparameters
        cont_cols (list): Continuous column names
        emb_szs (list): List of (vocab_size, emb_dim) tuples
        device (torch.device): Device to put model on
    
    Returns:
        TabularModel: Initialized model
    """
    try:
        model = TabularModel(
            emb_szs=emb_szs,
            n_cont=len(cont_cols) if cont_cols else 0,
            out_sz=1,
            layers=[
                params['layer_1_neurons'],
                params['layer_2_neurons'],
                params['layer_3_neurons'],
                params['layer_4_neurons']
            ],
            ps=[params['dropout']] * 4,
            emb_drop=params['embedding_dropout'],
            y_range=y_range,
            use_bn=params['use_bn'],
            bn_final=params['bn_final']
        ).to(device)

        logger.info("Model initialized successfully")
        logger.info(f"Model structure:\n{model}")
        
        return model
        
    except Exception as e:
        logger.error(f"Error initializing model: {str(e)}")
        raise
        
def train():
    """Main training function"""
    try:
        # Load hyperparameters and data
        params = load_hyperparameters()
        train_data, val_data, feature_config = load_training_data()

        # Set device and enable cuDNN benchmarking
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
        logger.info(f"Using device: {device}")

        # Extract features and target
        target_column = train_data.columns[0]  
        y_train = train_data[target_column]
        y_val = val_data[target_column]
        y_range = (0, y_train.max()*1.2)
        
        # Split features
        cat_features = feature_config.get('categorical', {})
        cont_features = feature_config.get('continuous', {})
        
        cat_cols = list(cat_features.keys())
        cont_cols = list(cont_features.keys())
        
        X_train_cat = train_data[cat_cols] if cat_cols else None
        X_train_cont = train_data[cont_cols] if cont_cols else None
        
        X_val_cat = val_data[cat_cols] if cat_cols else None
        X_val_cont = val_data[cont_cols] if cont_cols else None

        # Create embedding sizes
        emb_szs = [(len(cat_features[col]['classes']), 
                    min(50, (len(cat_features[col]['classes'])+1)//2))
                   for col in cat_cols] if cat_cols else []

        # Create datasets and dataloaders
        train_dataset = RegressionDataset(X_train_cat, X_train_cont, y_train)
        val_dataset = RegressionDataset(X_val_cat, X_val_cont, y_val)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=params['batch_size'],
            shuffle=True,
            num_workers=4
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=params['batch_size'],
            shuffle=False,
            num_workers=4
        )

        # Initialize model
        model = initialize_model(params, cont_cols, emb_szs, y_range, device) 

        # Initialize training
        criterion = nn.MSELoss()
        optimizer = optim.Adam(
            model.parameters(),
            lr=params['learning_rate'],
            weight_decay=params['weight_decay']
        )
        if params['optimizer'].lower() == "adam":
            optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])
        elif params['optimizer'].lower() == "sgd":
            optimizer = optim.SGD(model.parameters(), lr=params['learning_rate'], momentum=0.9, weight_decay=params['weight_decay'])
        elif params['optimizer'].lower() == "rmsprop":
            optimizer = optim.RMSprop(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])
        else:
            raise ValueError(f"Unknown optimizer: {params['optimizer']}")
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=2,
            verbose=True
        )
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        train_losses = []
        val_losses = []

        # Training loop
        for epoch in range(params['epochs']):
            logger.info(f"\nEpoch {epoch+1}/{params['epochs']}")
            logger.info(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Train and validate
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device, params)
            val_loss = validate(model, val_loader, criterion, device)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            logger.info(f'Training Loss: {train_loss:.6f}')
            logger.info(f'Validation Loss: {val_loss:.6f}')

            # Update learning rate
            scheduler.step(val_loss)

            # Early stopping check
            if val_loss < best_val_loss - params['early_stopping_min_delta']:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
                
                # Save current best model
                metrics = {
                    'best_validation_loss': float(best_val_loss),
                    'training_losses': [float(loss) for loss in train_losses],
                    'validation_losses': [float(loss) for loss in val_losses],
                    'epochs_trained': epoch + 1,
                    'final_learning_rate': float(optimizer.param_groups[0]['lr'])
                }
                save_model(model, metrics)
            else:
                patience_counter += 1
                logger.info(f"Early stopping counter: {patience_counter}/{params['early_stopping_patience']}")

                if patience_counter >= params['early_stopping_patience']:
                    logger.info("Early stopping triggered")
                    model.load_state_dict(best_model_state)
                    break

    except Exception as e:
        trc = traceback.format_exc()
        error_msg = f'Exception during training: {str(e)}\n{trc}'
        logger.error(error_msg)
        
        with open(os.path.join(output_path, 'failure'), 'w') as s:
            s.write(error_msg)
        
        raise

if __name__ == "__main__":
    train()
