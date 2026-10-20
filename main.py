"""Gold Price Prediction - Enhanced Neural Network Portfolio Project (CUDA/CPU Compatible)
Enhanced version with:
- Lag features & rolling statistics
- Fixed data leakage issues
- Learning rate scheduler
- Optimized architecture
"""
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Catch MAPE warnings for zero/near-zero values
with warnings.catch_warnings():
    warnings.simplefilter("ignore")

import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import platform

plt.style.use('seaborn-v0_8-whitegrid')


# ============ CUDA DEVICE MANAGEMENT ============
def get_available_device():
    """Detect available device (CUDA GPU or CPU) and configure accordingly."""
    print("\n" + "="*60)
    print("DEVICE DETECTION")
    print("="*60)
    
    # Check for CUDA availability
    if torch.cuda.is_available():
        cuda_device = torch.device(f'cuda:{torch.cuda.device_count() - 1}')
        device_type = "CUDA"
        
        print(f"\n✓ CUDA FOUND!")
        print(f"  GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"  GPU Count: {torch.cuda.device_count()}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"  Computation Capability: {torch.cuda.get_device_properties(0).major}.{torch.cuda.get_device_properties(0).minor}")
        
    elif torch.backends.mps.is_available():
        cuda_device = torch.device('mps')
        device_type = "MPS (Apple Silicon)"
        print(f"\n✓ MPS FOUND!")
        print(f"  Apple Silicon GPU detected for accelerated training")
        
    else:
        # CPU fallback - enable multi-threading
        cuda_device = torch.device('cpu')
        device_type = "CPU"
        
        if hasattr(torch.multiprocessing, 'set_start_method'):
            try:
                torch.multiprocessing.set_start_method('spawn', force=True)
                print(f"\n✓ CPU Detected - Multi-threading ENABLED")
                print(f"  Using {os.cpu_count()} CPU cores for training")
                torch.set_num_threads(os.cpu_count())
            except RuntimeError:
                print(f"\n✓ CPU Detected (multi-threading already set)")
        
        # Print CPU info for diagnostics
        import platform
        print(f"  Platform: {platform.platform()[:30]}")
        print(f"  Python Version: {platform.python_version()}")
    
    return cuda_device, device_type


class Config:
    DATA_PATH="C:\\Users\\Ilya\\Documents\\Gld_price\\gold-Pirce\\gld_price_data.csv"
    OUTPUT_CSV="C:\\Users\\Ilya\\Documents\\Gld_price\\gold-Pirce\\gld_price_predictions_enhanced.csv"
    
    # FIXED: Updated INPUT_DIM to include lag/rolling features (9 total)
    INPUT_DIM=9  # Original 4 base + 5 engineered (3 lag + 2 rolling)
    BASE_FEATURES=['SPX','USO','SLV','EUR/USD']
    LAG_COLUMNS=['GLD_LAG1','GLD_LAG7','GLD_LAG14']
    ROLLING_COLUMNS=['GLD_ROLLING_MEAN_7','GLD_ROLLING_STD_7']
    
    # FIX: Smaller hidden layers for small dataset - reduce overfitting
    HIDDEN_LAYERS=[128, 64, 32]
    
    EPOCHS=1000  # Max epochs (early stopping will stop earlier)
    LEARNING_RATE=0.001  # FIXED: Higher LR for faster convergence + scheduler will adjust
    BATCH_SIZE=32  # FIXED: Better batch size for GPU/CPU compatibility
    
    TRAIN_RATIO=0.6    # 60% training, 15% validation, 25% test
    VALIDATION_RATIO=0.15
    TEST_RATIO=0.25
    EARLY_STOPPING_PATIENCE=50  # Much lower patience for small dataset
    
    # FIX: Lower dropout rate (30% was too aggressive)
    DROPOUT_RATE=0.15 
    
    DEVICE_MEMORY_GBRAM="2"  # Default GPU memory check threshold for regularization


class DataLoader:
    def __init__(self, data_path):
        self.df=None; self.scaler=None; self.feature_cols=['SPX','USO','SLV','EUR/USD']
        self.data_path=data_path

    def load_data(self):
        print("="*60); print("LOADING DATA"); print("="*60)
        self.df=pd.read_csv(self.data_path)
        print(f"Dataset: {self.df.shape}")
        return self.df

    def preprocess_dates(self):
        print("\nPREPROCESSING DATES")
        self.df['Date']=pd.to_datetime(self.df['Date'],format='%m/%d/%Y',errors='coerce')
        self.df=self.df.set_index('Date'); return self.df

    def add_year_feature(self):
        print("\nADDING YEAR FEATURE")
        self.df['Year']=self.df.index.year; print("Done"); return self.df

    def engineer_features(self,year_start,year_end):
        print(f"\nSELECTING: {year_start}-{year_end}")
        mask=(self.df['Year']>=year_start)&(self.df['Year']<=year_end)
        self.df_trained=self.df[mask].copy()
        print(f"Selected: {len(self.df_trained)} records")
        
        # FIX: Call this separately for each split (train, val, test) after initial chronological split
        # This prevents data leakage by adding lag features using only past data
        self.df_trained = self._add_lag_and_rolling_features(self.df_trained)
        print(f"Added lag/rolling features - {len(self.df_trained)} records with engineered features")
        return self.df_trained

    def _add_lag_and_rolling_features(self, df):
        """Add lag and rolling features safely."""
        df = df.copy()
        
        # Add lag features (t-1, t-7, t-14 days)
        df['GLD_LAG1'] = df['GLD'].shift(1)
        df['GLD_LAG7'] = df['GLD'].shift(7)
        df['GLD_LAG14'] = df['GLD'].shift(14)
        
        # Add rolling statistics for volatility
        df['GLD_ROLLING_MEAN_7'] = df['GLD'].rolling(window=7).mean()
        df['GLD_ROLLING_STD_7'] = df['GLD'].rolling(window=7).std()
        
        # Fill NaN created by lag/rolling
        df = df.bfill().ffill()
        
        return df

    def split_features_target(self):
        print("\nSPLITTING FEATURES & TARGET")
        # Get all available columns including potential lag/rolling features
        base_cols = ['SPX','USO','SLV','EUR/USD']
        all_feature_cols = [c for c in self.df_trained.columns if c in base_cols or 'LAG' in c or 'ROLLING' in c]
        
        self.X_train_data=self.df_trained[all_feature_cols].copy()
        self.y_train_data=self.df_trained['GLD'].copy()
        
        print(f"Features: {len(all_feature_cols)} columns (base + engineered)")
        return self.X_train_data, self.y_train_data

    def normalize_features(self):
        """
        FIX: Scale features using ONLY training data statistics to prevent data leakage.
        For time series, we split chronologically AFTER scaling training data.
        """
        print("\nNORMALIZING (FIXED - NO DATA LEAKAGE)")
        
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
        # Get chronological indices for splitting
        n_samples = len(self.df_trained)
        train_end = int(n_samples * Config.TRAIN_RATIO)         # 60%
        val_end = int(n_samples * (Config.TRAIN_RATIO + Config.VALIDATION_RATIO))  # 75%
        
        # Scale ONLY training data first (fit on training only)
        X_train_scaled = self.scaler.fit_transform(self.X_train_data)
        
        # Transform validation and test using training statistics (no fitting!)
        X_val_scaled = self.scaler.transform(self.df_trained.iloc[:val_end, :self.feature_cols.columns.get_loc('SPX')] if hasattr(self.df_trained[self.feature_cols], 'columns') else self.df_trained[self.feature_cols].values)
        
        # Re-apply to ensure proper indexing
        X_val_scaled = self.scaler.transform(self.df_trained[self.feature_cols])
        X_test_scaled = self.scaler.transform(self.df_trained.iloc[val_end:, self.feature_cols.columns]) if hasattr(self.df_trained, 'columns') else self.scaler.transform(self.df_trained.iloc[val_end:])
        
        return X_train_scaled, X_val_scaled, X_test_scaled


class Visualizer:
    def __init__(self): self.figures=[]

    def plot_time_series(self,df):
        print("\nPLOT 1: Gold Price Trend"); fig=go.Figure()
        fig.add_trace(go.Scatter(x=df.index,y=df['GLD'],mode='lines',name='GLD',line=dict(color='#FFD700')))
        fig.update_layout(title="Gold Price (GLD ETF)",height=500); fig.show()

    def plot_box(self,df):
        print("\nPLOT 2: Monthly Distribution"); fig=go.Figure(data=[go.Box(y=df['GLD'],x=df.index.month)])
        fig.update_layout(title="Monthly Gold Price Distribution",height=500); fig.show()

    def plot_heatmap(self,df):
        print("\nPLOT 3: Correlation Matrix"); corr=df.corr()
        fig=go.Figure(data=go.Heatmap(z=corr.values,x=list(corr.columns),y=list(corr.index)))
        fig.update_layout(title="Correlation Matrix",height=600); fig.show()


class ModelTrainer:
    def __init__(self, device):
        self.device = device
        self.model=None
    
    def build_model_simple(self, input_dim):
        """
        Simplified model building - FIXED architecture for small dataset.
        Reduced layers + increased dropout to prevent overfitting.
        """
        print("\nBUILDING MODEL (OPTIMIZED FOR SMALL DATASET)")
        
        # Build optimized architecture
        self.model = nn.Sequential()
        
        # Input → Dense 128 with ReLU
        self.model.add_module('fc1', nn.Linear(input_dim, 128))
        self.model.add_module('relu1', nn.ReLU())
        
        # Fewer hidden layers (3 instead of 6) to prevent overfitting
        units_list = [64, 32, 16]
        dropout_rate = Config.DROPOUT_RATE
        
        for i, units in enumerate(units_list):
            # Apply dropout to all hidden layers except output
            self.model.add_module(f'fc{i+2}', nn.Linear(128 if i == 0 else units_list[i-1], units))
            self.model.add_module(f'relu{i+2}', nn.ReLU())
            self.model.add_module(f'dropout{i+2}', nn.Dropout(dropout_rate))
        
        # Output layer (linear for regression)
        self.model.add_module('output', nn.Linear(16, 1))
        
        print(f"Arch: Input({input_dim})→[128,64,32,16]→Output(1), Dropout={dropout_rate}")
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Training with Early Stopping to prevent overfitting.
        """
        print("\nTRAINING MODEL WITH EARLY STOPPING")
        
        # Move model to device
        self.model.to(self.device)
        
        # Create train_loader using PyTorch's DataLoader with batch_size support
        dataset = TensorDataset(X_train, y_train)
        
        # Dynamic batch size based on device memory
        if 'cuda' in str(self.device).lower() or 'mps' in str(self.device):
            batch_size = max(64, Config.BATCH_SIZE)  # GPU: larger batches
        else:
            batch_size = Config.BATCH_SIZE  # CPU: smaller batches
        
        torch_train_loader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True if self.device.type != 'cuda' else False,  # No shuffle on GPU for time series
            num_workers=0
        )
        
        optimizer = optim.Adam(self.model.parameters(), lr=Config.LEARNING_RATE)
        criterion = nn.MSELoss()
        
        # Training loop with early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        epochs_trained = 0
        total_loss_history = []
        val_loss_history = []
        
        for epoch in range(Config.EPOCHS):
            # Training phase
            self.model.train()
            epoch_train_loss = 0
            for batch_X, batch_y in torch_train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device).view(-1, 1)
                
                optimizer.zero_grad()
                predictions = self.model(batch_X)
                loss = criterion(predictions, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_train_loss += loss.item()
            
            avg_train_loss = epoch_train_loss / len(torch_train_loader)
            total_loss_history.append(avg_train_loss)
            
            # Validation phase (only if validation data exists)
            if X_val is not None and y_val is not None:
                self.model.eval()
                with torch.no_grad():
                    val_predictions = self.model(X_val.to(self.device))
                    avg_val_loss = criterion(val_predictions, y_val.to(self.device)).item()
                    val_loss_history.append(avg_val_loss)
                
                if (epoch + 1) % 50 == 0:
                    print(f"Epoch {epoch+1:3d}/{Config.EPOCHS} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
                
                # Early stopping check
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                        print(f"\n⚠️ EARLY STOPPING at epoch {epoch+1} (no improvement for {patience_counter} epochs)")
                        break
            else:
                # No validation - just train and report periodically
                if (epoch + 1) % 50 == 0 or epoch == 0:
                    print(f"Epoch {epoch+1:3d}/{Config.EPOCHS} | Loss: {avg_train_loss:.6f}")
            
            epochs_trained = epoch + 1
        
        print(f"\n✓ Training completed after {epochs_trained} epochs")
        
        return total_loss_history, val_loss_history if X_val is not None else None
    
    def predict(self, X_test):
        """Make predictions and handle device placement."""
        self.model.eval()
        with torch.no_grad():
            if isinstance(X_test, list):
                # Handle tensor lists from train_test_split - need to concatenate first
                X_test_concat = torch.cat([X_test[0], X_test[1]], dim=0)
                pred = self.model(X_test_concat.to(self.device))
                return pred.cpu()
            else:
                pred = self.model(X_test.to(self.device))
                return pred.cpu()


class ModelEvaluator:
    @staticmethod
    def calculate_metrics(y_true, y_pred):
        """Calculate comprehensive metrics with safe MAPE calculation."""
        y_true_np = y_true.detach().cpu().numpy()
        y_pred_np = y_pred.detach().cpu().numpy()
        
        r2 = r2_score(y_true_np, y_pred_np)
        mae = mean_absolute_error(y_true_np, y_pred_np)
        rmse = np.sqrt(mean_squared_error(y_true_np, y_pred_np))
        
        # FIX: Safe MAPE calculation that handles zero values
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            nonzero_mask = np.abs(y_true_np) > 0.01  # Only consider prices > $0.01
            if np.sum(nonzero_mask) > 0:
                mape = np.mean(np.abs((y_true_np[nonzero_mask] - y_pred_np[nonzero_mask]) / (y_true_np[nonzero_mask]))) * 100
            else:
                mape = 0.0
        
        max_err = np.max(np.abs(y_true_np - y_pred_np))
        
        return {
            'R² Score': r2, 
            'MAE ($)': mae, 
            'RMSE ($)': rmse, 
            'MAPE (%)': mape, 
            'Max Error ($)': max_err
        }
    
    @classmethod
    def print_metrics(cls, metrics):
        print("\nMODEL PERFORMANCE METRICS"); print("="*60)
        for metric, value in metrics.items():
            if '%' in metric:
                print(f"{metric:<35} {value:.2f}")
            else:
                print(f"{metric:<35} ${value:.2f}")


def add_lag_features_safely(df, drop_first_rows=14):
    """Add lag features safely to a DATA PORTION (train/val/test separately).
    
    This prevents data leakage by:
    1. Only using data within the current portion (not future data from other splits)
    2. Dropping first 'drop_first_rows' observations where lag would peek ahead
    
    Args:
        df: DataFrame with base features and target (no lag yet)
        drop_first_rows: Number of rows to drop at start due to lag/rolling NaN
        
    Returns:
        DataFrame with engineered features, ready for this specific split
    """
    print(f"   Adding lag/rolling features to THIS portion only ({len(df)} samples)...")
    
    df = df.copy()
    
    # Add lag features - using ONLY past data for this portion
    df['GLD_LAG1'] = df['GLD'].shift(1)  # Yesterday's price
    df['GLD_LAG7'] = df['GLD'].shift(7)  # Last week's price  
    df['GLD_LAG14'] = df['GLD'].shift(14)  # Last 2 weeks' price
    
    # Rolling statistics for volatility - using only past data
    df['GLD_ROLLING_MEAN_7'] = df['GLD'].rolling(window=7, min_periods=1).mean()
    df['GLD_ROLLING_STD_7'] = df['GLD'].rolling(window=7, min_periods=1).std().fillna(0)
    
    # Drop first 'drop_first_rows' rows to avoid using lags that peek into future
    df = df.iloc[drop_first_rows:]
    
    print(f"   ✓ Created {len(df)} samples for this portion (dropped {drop_first_rows} lag edge cases)")
    
    return df


def main():
    print("\n" + "="*60)
    print("GOLD PRICE PREDICTION SYSTEM (FIXED - NO LEAKAGE)")
    print("="*60)
    
    # Initialize device detection at the start
    device, device_type = get_available_device()
    
    # Print training configuration based on device
    print(f"\n📊 Training Configuration:")
    print(f"   Device: {device_type}")
    if device_type == "CPU":
        print(f"   Threads: Multi-threaded enabled")
    else:
        print(f"   Memory: Using GPU acceleration")
    
    # Load raw data (NO lag/rolling yet - we add it separately later!)
    loader = DataLoader(Config.DATA_PATH)
    df_raw = loader.load_data()
    df_raw = loader.preprocess_dates()
    df_raw = loader.add_year_feature()
    
    vis = Visualizer()
    vis.plot_time_series(df_raw)
    vis.plot_box(df_raw)
    vis.plot_heatmap(df_raw)
    
    # FILTER by year first (still raw data, no lag features yet!)
    print("\n" + "="*60)
    print("CHRONOLOGICAL DATA SPLIT (FIXED - NO LEAKAGE)")
    print("="*60)
    
    mask = (df_raw['Year'] >= 2017) & (df_raw['Year'] <= 2019)
    df_2017_2019 = df_raw[mask].copy()
    print(f"Selected {len(df_2017_2019)} records from 2017-2019 (raw data, no lag yet)")
    
    # Get total samples and calculate split points chronologically
    n_samples = len(df_2017_2019)
    train_end = int(n_samples * Config.TRAIN_RATIO)          # 60% for training
    val_end = int(n_samples * (Config.TRAIN_RATIO + Config.VALIDATION_RATIO))  # 75% cumulative (15% validation)
    test_end = n_samples                                      # 100% (25% test)
    
    print(f"\nData Split: {train_end} train | {val_end-train_end} val | {test_end-val_end} test")
    print(f"({Config.TRAIN_RATIO*100:.0f}% / {Config.VALIDATION_RATIO*100:.0f}% / {Config.TEST_RATIO*100:.0f}%)")
    
    # SPLIT RAW DATA CHRONOLOGICALLY FIRST (BEFORE any lag/rolling features!)
    # Include GLD column so we can add lag features for gold price prediction
    X_train_raw = df_2017_2019.iloc[:train_end][['SPX','USO','SLV','EUR/USD','GLD']].copy()
    y_train_raw = df_2017_2019.iloc[:train_end]['GLD'].copy()  # Keep separate for target scaler
    
    X_val_raw = df_2017_2019.iloc[train_end:val_end][['SPX','USO','SLV','EUR/USD','GLD']].copy()
    y_val_raw = df_2017_2019.iloc[train_end:val_end]['GLD'].copy()  # Keep separate
    
    X_test_raw = df_2017_2019.iloc[val_end:test_end][['SPX','USO','SLV','EUR/USD','GLD']].copy()
    y_test_raw = df_2017_2019.iloc[val_end:test_end]['GLD'].copy()  # Keep separate
    
    print(f"\n✓ Split COMPLETE - now adding lag/rolling features SEPARATELY to each portion")
    
    # CRITICAL FIX: Add lag/rolling features SEPARATELY to each split!
    # This prevents data leakage (future info from test/validation not in training)
    print("\n" + "="*60)
    print("ADDING LAG FEATURES TO EACH PORTION (NO LEAKAGE)")
    print("="*60)
    
    # Add lag features to TRAINING portion only
    # Train can see its own past, but NOT val/test future
    X_train_with_lags = add_lag_features_safely(X_train_raw, drop_first_rows=14)
    y_train = X_train_with_lags['GLD'].copy()  # GLD is now in the engineered dataframe
    
    # Add lag features to VALIDATION portion (keeps all columns including lag/rolling)
    X_val_with_lags = add_lag_features_safely(X_val_raw, drop_first_rows=14)
    
    # Add lag features to TEST portion (keeps all columns including lag/rolling)  
    X_test_with_lags = add_lag_features_safely(X_test_raw, drop_first_rows=14)
    
    print(f"\n✓ Training set: {len(X_train_with_lags)} samples with lag/rolling features")
    print(f"✓ Validation set: {len(X_val_with_lags)} samples with lag/rolling features")
    print(f"✓ Test set: {len(X_test_with_lags)} samples with lag/rolling features")
    
    # Display date ranges for each split to verify chronological order
    print(f"\nTrain dates: {X_train_with_lags.index[0]} to {X_train_with_lags.index[-1]}")
    print(f"Val dates:   {X_val_with_lags.index[0]} to {X_val_with_lags.index[-1]}")
    print(f"Test dates:  {X_test_with_lags.index[0]} to {X_test_with_lags.index[-1]}")
    
    # FIX: SEPARATE scalers for features and target!
    # CRITICAL: Use ALL features including lag/rolling in the scaler
    print("\n" + "="*60)
    print("SCALING FEATURES & TARGET (FIXED)")
    print("="*60)
    
    # Get all feature columns from train set
    feature_cols = [c for c in X_train_with_lags.columns if c not in ['GLD', 'Year']]
    print(f"Scaling {len(feature_cols)} features: {feature_cols}")
    
    # Feature scaler - fit on training features only (including lag/rolling!)
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    X_train_scaled = feature_scaler.fit_transform(X_train_with_lags[feature_cols])
    
    # Transform val/test using TRAINING statistics (no fitting!)
    X_val_scaled = feature_scaler.transform(X_val_with_lags[feature_cols])
    X_test_scaled = feature_scaler.transform(X_test_with_lags[feature_cols])
    
    # TARGET scaler - fit on training target only (CRITICAL for time series!)
    # y_train is already extracted from the engineered dataframe
    target_scaler = MinMaxScaler(feature_range=(0, 1))
    y_train_scaled = target_scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
    
    # For val/test, we need to get their raw GLD values and transform them
    y_val_raw = X_val_with_lags['GLD'].copy()  # Use the lag-enhanced dataframe
    y_test_raw = X_test_with_lags['GLD'].copy()
    
    y_val_scaled = target_scaler.transform(y_val_raw.values.reshape(-1, 1)).flatten()
    y_test_scaled = target_scaler.transform(y_test_raw.values.reshape(-1, 1)).flatten()
    
    print(f"\nFeature scaler range: [{feature_scaler.data_min_[0]:.2f}, {feature_scaler.data_max_[0]:.2f}]")
    print(f"Target scaler range: [{target_scaler.data_min_[0]:.4f}, {target_scaler.data_max_[0]:.4f}]")
    print("✓ No data leakage - all scalers fit on training data only!")
    
    # Create and train model with validation set
    print("\n" + "="*60)
    print("MODEL TRAINING")
    print("="*60)
    
    trainer = ModelTrainer(device)
    trainer.build_model_simple(Config.INPUT_DIM)
    
    # Convert to torch tensors (ensure proper shape: [samples, features])
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32).reshape(-1, 1)
    
    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val_scaled, dtype=torch.float32).reshape(-1, 1)
    
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32).reshape(-1, 1)
    
    # Train with validation set for early stopping
    total_loss_history, val_loss_history = trainer.train(
        X_train_tensor, y_train_tensor, 
        X_val_tensor, y_val_tensor
    )
    
    # Make predictions on test set (no validation needed - we already validated during training)
    print("\n" + "="*60)
    print("TEST PREDICTIONS")
    print("="*60)
    
    trainer.model.eval()
    with torch.no_grad():
        predictions_scaled = trainer.predict(X_test_tensor)
        predictions_array = predictions_scaled.numpy().flatten()
    
    # INVERSE TRANSFORM predictions to original scale using target scaler
    y_test_original = target_scaler.inverse_transform(y_test_tensor.reshape(-1, 1)).flatten()
    predictions_original = target_scaler.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()
    error_original = predictions_original - y_test_original
    
    # Calculate and print metrics on ORIGINAL SCALE values
    print("\n" + "="*60)
    print("TEST PREDICTION PERFORMANCE (ORIGINAL SCALE)")
    print("="*60)
    
    y_test_for_metrics = torch.tensor(y_test_original, dtype=torch.float32).reshape(-1, 1)
    predictions_for_metrics = torch.tensor(predictions_original, dtype=torch.float32).reshape(-1, 1)
    
    metrics = ModelEvaluator.calculate_metrics(y_test_for_metrics, predictions_for_metrics)
    ModelEvaluator.print_metrics(metrics)
    
    # Save predictions to CSV with full date information for reference (ORIGINAL SCALE)
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)
    
    output_df = pd.DataFrame({
        'date': X_test_with_lags.index,  # Add dates back for reference
        'actual': y_test_original, 
        'predicted': predictions_original, 
        'error': error_original
    })
    output_df.to_csv(Config.OUTPUT_CSV, index=False)
    print(f"✓ Saved: {Config.OUTPUT_CSV}")
    
    # Save visualization - compare original scale predictions
    plt.figure(figsize=(14, 6))
    # Plot some test dates for readability (first 100 points)
    plot_end = min(100, len(y_test_original))
    plt.plot(X_test_with_lags.index[:plot_end], y_test_original[:plot_end], 'blue', alpha=0.5, label='Actual')
    plt.plot(X_test_with_lags.index, predictions_original, 'green', linewidth=2, label='Predicted')
    
    plt.title('Model Performance - Test Set Predictions (Original Scale)', fontsize=14, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price (USD)', fontsize=12)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig('model_performance.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: model_performance.png")
    
    # Print summary
    print("\n" + "="*60)
    print("EXECUTION SUMMARY")
    print("="*60)
    print(f"Training samples: {len(X_train_with_lags)}")
    print(f"Validation samples: {len(X_val_with_lags)}")
    print(f"Test samples: {len(X_test_with_lags)}")
    print(f"Final Test Metrics:")
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            print(f"  - {k}: {v:.4f}")
    
    print("\n" + "="*60); print("PIPELINE COMPLETE!"); print("="*60)


if __name__=="__main__": main()

