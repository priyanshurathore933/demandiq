import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import pickle
import os

class HybridForecastingModel:
    """
    Hybrid Forecasting Model combining LSTM and XGBoost
    LSTM captures temporal patterns, XGBoost handles feature interactions
    """
    
    def __init__(self, lookback_period=30, forecast_horizon=30):
        self.lookback_period = lookback_period
        self.forecast_horizon = forecast_horizon
        self.lstm_model = None
        self.xgb_model = None
        self.scaler = MinMaxScaler()
        self.feature_scaler = MinMaxScaler()
        
    def create_lstm_model(self, input_shape):
        """Create LSTM architecture"""
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def prepare_sequences(self, data, sentiment_features=None):
        """Prepare sequences for LSTM training"""
        X, y = [], []
        
        for i in range(len(data) - self.lookback_period - self.forecast_horizon + 1):
            # Get sequence
            seq = data[i:i + self.lookback_period]
            
            # Add sentiment features if available
            if sentiment_features is not None:
                sentiment_seq = sentiment_features[i:i + self.lookback_period]
                seq = np.column_stack([seq, sentiment_seq])
            
            X.append(seq)
            y.append(data[i + self.lookback_period:i + self.lookback_period + self.forecast_horizon])
        
        return np.array(X), np.array(y)
    
    def extract_features(self, df, sentiment_data=None):
        """Extract features for XGBoost"""
        features = pd.DataFrame()
        
        # Time-based features
        df['date'] = pd.to_datetime(df['date'])
        features['day_of_week'] = df['date'].dt.dayofweek
        features['day_of_month'] = df['date'].dt.day
        features['month'] = df['date'].dt.month
        features['quarter'] = df['date'].dt.quarter
        features['is_weekend'] = (df['date'].dt.dayofweek >= 5).astype(int)
        
        # Lag features
        for lag in [7, 14, 30]:
            features[f'lag_{lag}'] = df['quantity'].shift(lag)
        
        # Rolling statistics
        for window in [7, 14, 30]:
            features[f'rolling_mean_{window}'] = df['quantity'].rolling(window=window).mean()
            features[f'rolling_std_{window}'] = df['quantity'].rolling(window=window).std()
        
        # Price features
        if 'price' in df.columns:
            features['price'] = df['price']
            features['price_change'] = df['price'].pct_change()
        
        # Sentiment features
        if sentiment_data is not None:
            features['sentiment_score'] = sentiment_data.get('sentiment_score', 0)
            features['mentions'] = sentiment_data.get('mentions', 0)
            features['engagement'] = sentiment_data.get('engagement', 0)
        
        # Fill NaN values
        features = features.fillna(method='bfill').fillna(0)
        
        return features
    
    def train_lstm(self, data, sentiment_features=None, epochs=50, batch_size=32):
        """Train LSTM model"""
        print("🔄 Training LSTM model...")
        
        # Normalize data
        scaled_data = self.scaler.fit_transform(data.reshape(-1, 1))
        
        # Prepare sequences
        X, y = self.prepare_sequences(scaled_data.flatten(), sentiment_features)
        
        # Reshape for LSTM [samples, timesteps, features]
        if sentiment_features is not None:
            X = X.reshape(X.shape[0], X.shape[1], 2)
        else:
            X = X.reshape(X.shape[0], X.shape[1], 1)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        # Create model
        self.lstm_model = self.create_lstm_model((X.shape[1], X.shape[2]))
        
        # Callbacks
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        # Train
        history = self.lstm_model.fit(
            X_train, y_train[:, 0],  # Predict only next day
            validation_data=(X_val, y_val[:, 0]),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop],
            verbose=1
        )
        
        print(f"✅ LSTM training complete. Final loss: {history.history['loss'][-1]:.4f}")
        
        return history
    
    def train_xgboost(self, df, sentiment_data=None):
        """Train XGBoost model"""
        print("🔄 Training XGBoost model...")
        
        # Extract features
        features = self.extract_features(df, sentiment_data)
        X = features.values
        y = df['quantity'].values
        
        # Remove rows with NaN
        mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        X = X[mask]
        y = y[mask]
        
        # Scale features
        X = self.feature_scaler.fit_transform(X)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        # Train XGBoost
        self.xgb_model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        
        self.xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=20,
            verbose=False
        )
        
        # Evaluate
        val_score = self.xgb_model.score(X_val, y_val)
        print(f"✅ XGBoost training complete. R² score: {val_score:.4f}")
        
        return val_score
    
    def predict(self, data, sentiment_features=None, df=None, sentiment_data=None):
        """Make predictions using ensemble of LSTM and XGBoost"""
        predictions = []
        
        # LSTM prediction
        if self.lstm_model is not None:
            scaled_data = self.scaler.transform(data.reshape(-1, 1))
            
            if sentiment_features is not None:
                input_seq = np.column_stack([
                    scaled_data[-self.lookback_period:].flatten(),
                    sentiment_features[-self.lookback_period:]
                ])
                input_seq = input_seq.reshape(1, self.lookback_period, 2)
            else:
                input_seq = scaled_data[-self.lookback_period:].reshape(1, self.lookback_period, 1)
            
            lstm_pred = self.lstm_model.predict(input_seq, verbose=0)
            lstm_pred = self.scaler.inverse_transform(lstm_pred.reshape(-1, 1)).flatten()
            predictions.append(lstm_pred)
        
        # XGBoost prediction
        if self.xgb_model is not None and df is not None:
            features = self.extract_features(df, sentiment_data)
            X = self.feature_scaler.transform(features.values[-1:])
            xgb_pred = self.xgb_model.predict(X)
            predictions.append(xgb_pred)
        
        # Ensemble: weighted average
        if len(predictions) == 2:
            final_pred = 0.6 * predictions[0] + 0.4 * predictions[1]
        elif len(predictions) == 1:
            final_pred = predictions[0]
        else:
            final_pred = np.array([data[-1]])
        
        return final_pred
    
    def forecast_with_confidence(self, data, sentiment_features=None, df=None, sentiment_data=None, steps=30):
        """Generate forecast with confidence intervals"""
        predictions = []
        lower_bounds = []
        upper_bounds = []
        
        current_data = data.copy()
        
        for step in range(steps):
            # Make prediction
            pred = self.predict(current_data, sentiment_features, df, sentiment_data)
            
            # Calculate confidence intervals (using historical std)
            std = np.std(data[-30:]) if len(data) >= 30 else np.std(data)
            lower = pred - 1.96 * std
            upper = pred + 1.96 * std
            
            predictions.append(pred[0])
            lower_bounds.append(max(0, lower[0]))
            upper_bounds.append(upper[0])
            
            # Update data for next prediction
            current_data = np.append(current_data, pred)
        
        return np.array(predictions), np.array(lower_bounds), np.array(upper_bounds)
    
    def save_model(self, path='models/'):
        """Save trained models"""
        os.makedirs(path, exist_ok=True)
        
        if self.lstm_model is not None:
            self.lstm_model.save(os.path.join(path, 'lstm_model.h5'))
        
        if self.xgb_model is not None:
            with open(os.path.join(path, 'xgb_model.pkl'), 'wb') as f:
                pickle.dump(self.xgb_model, f)
        
        with open(os.path.join(path, 'scalers.pkl'), 'wb') as f:
            pickle.dump({
                'scaler': self.scaler,
                'feature_scaler': self.feature_scaler
            }, f)
        
        print(f"✅ Models saved to {path}")
    
    def load_model(self, path='models/'):
        """Load trained models"""
        lstm_path = os.path.join(path, 'lstm_model.h5')
        xgb_path = os.path.join(path, 'xgb_model.pkl')
        scalers_path = os.path.join(path, 'scalers.pkl')
        
        if os.path.exists(lstm_path):
            self.lstm_model = keras.models.load_model(lstm_path)
            print("✅ LSTM model loaded")
        
        if os.path.exists(xgb_path):
            with open(xgb_path, 'rb') as f:
                self.xgb_model = pickle.load(f)
            print("✅ XGBoost model loaded")
        
        if os.path.exists(scalers_path):
            with open(scalers_path, 'rb') as f:
                scalers = pickle.load(f)
                self.scaler = scalers['scaler']
                self.feature_scaler = scalers['feature_scaler']
            print("✅ Scalers loaded")
        
        return self

# Example usage
if __name__ == "__main__":
    # Sample data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=365, freq='D')
    quantities = 1000 + 200 * np.sin(np.arange(365) * 2 * np.pi / 365) + np.random.normal(0, 50, 365)
    
    df = pd.DataFrame({
        'date': dates,
        'quantity': quantities,
        'price': 29.99
    })
    
    # Initialize and train model
    model = HybridForecastingModel(lookback_period=30, forecast_horizon=30)
    
    # Train LSTM
    model.train_lstm(df['quantity'].values, epochs=20, batch_size=16)
    
    # Train XGBoost
    model.train_xgboost(df)
    
    # Make forecast
    predictions, lower, upper = model.forecast_with_confidence(
        df['quantity'].values,
        df=df,
        steps=30
    )
    
    print(f"\n📈 30-day Forecast:")
    print(f"Mean: {predictions.mean():.0f}")
    print(f"Range: {predictions.min():.0f} - {predictions.max():.0f}")
    
    # Save model
    model.save_model()