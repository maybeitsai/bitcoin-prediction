#!/usr/bin/env python
# coding: utf-8

# # Bitcoin Price Prediction using Deep Learning
# 
# - Author: Harry Mardika
# - Date: 3 May 2025
# 
# ## Introduction
# 
# ### Project Objective
# The primary goal of this project is to predict the future price of Bitcoin (BTC) using historical daily data. Accurate Bitcoin price prediction is crucial for investors, traders, and financial institutions to make informed decisions, manage risk, and potentially optimize trading strategies.
# 
# ### Importance of Bitcoin Forecasting
# Bitcoin is known for its high volatility and complex price dynamics, influenced by a myriad of factors including market sentiment, regulatory news, macroeconomic trends, and technological developments. Forecasting its price is challenging but offers significant potential rewards. Advanced time series models can help capture the inherent patterns and dependencies in the price movements.
# 
# ### Models Used
# We will implement and compare two deep learning models known for their effectiveness in sequence modeling:
# 1.  **LSTM (Long Short-Term Memory):** A type of Recurrent Neural Network (RNN) capable of learning long-range dependencies, making it suitable for time series data.
# 2.  **CNN-LSTM Hybrid:** This model combines Convolutional Neural Networks (CNNs) to extract spatial hierarchies or local patterns within time steps (treating sequences like 1D images) and LSTMs to model the temporal dependencies between these extracted features.
# 
# ### Dataset
# The dataset comprises daily Bitcoin data from January 1st, 2016, to May 2nd, 2025 (as requested, although data beyond the current date would be synthetic or from a specific source not generally available - using a placeholder endpoint like a recent date, e.g., early 2024, is more realistic. For this example, we will assume the data up to a recent date is available and proceed). It includes columns: "Tanggal" (Date), "Terakhir" (Close), "Pembukaan" (Open), "Tertinggi" (High), "Terendah" (Low), "Vol." (Volume), and "Perubahan%" (Change %).
# 
# Let's begin by preparing the data.

# ## 1. Data Preparation

# ### Import Libraries
# Import necessary libraries for data manipulation, visualization, preprocessing, and modeling.

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten, Input, TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import AdamW
import warnings

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid') # Use a visually appealing style


# ### Load Data
# Load the dataset from a CSV file. Parse the date column and rename columns to English for clarity.
# 

# In[2]:


df = pd.read_csv('/kaggle/input/bitcoin-2-may-2025/btc_2_may_25.csv')
print("Dataset loaded successfully.")
print("Original columns:", df.columns.tolist())


# In[5]:


# Rename columns to English
column_mapping = {
    "Tanggal": "Date",
    "Terakhir": "Close",
    "Pembukaan": "Open",
    "Tertinggi": "High",
    "Terendah": "Low",
    "Vol.": "Volume",
    "Perubahan%": "Change_Percent"
}
df.rename(columns=column_mapping, inplace=True)


# ### Data Cleaning and Type Conversion
# Convert columns to their appropriate data types. Handle specific formats for 'Volume' and 'Change_Percent'.

# In[6]:


# Convert 'Date' to datetime objects
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')


# In[7]:


# Function to convert volume strings (e.g., '123.4K', '5.6M') to numeric
def convert_volume_to_numeric(volume_str):
    if isinstance(volume_str, (int, float)):
        return volume_str
    volume_str = str(volume_str).strip().upper()
    if volume_str == '-' or volume_str == '':
        return np.nan
    multiplier = 1
    if 'K' in volume_str:
        multiplier = 1_000
        volume_str = volume_str.replace('K', '')
    elif 'M' in volume_str:
        multiplier = 1_000_000
        volume_str = volume_str.replace('M', '')
    try:
        # Handle potential comma as decimal separator if needed, assuming '.' is decimal here
        volume_str = volume_str.replace(',', '')
        return float(volume_str) * multiplier
    except ValueError:
        return np.nan


# In[10]:


# Convert 'Volume' to numeric
df['Volume'] = df['Volume'].apply(convert_volume_to_numeric)

# Convert 'Change_Percent' to numeric (float)
df['Change_Percent'] = df['Change_Percent'].astype(str).str.replace('%', '').str.replace(',', '.').astype(float) / 100.0

# Convert price columns to numeric, coercing errors to NaN
price_cols = ['Close', 'Open', 'High', 'Low']
for col in price_cols:
     # Handle potential thousands separators (like '.') if dataset uses it
    if df[col].dtype == 'object':
         df[col] = df[col].str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
    df[col] = pd.to_numeric(df[col], errors='coerce')


# ### Handle Missing Values
# Check for missing values and apply an appropriate strategy (e.g., forward fill).

# In[11]:


print("\nMissing values before handling:")
print(df.isnull().sum())


# In[21]:


# Handle NaNs - Forward fill is suitable for time series price data
df.fillna(method='ffill', inplace=True)
# If any NaNs remain at the beginning, backfill them
df.fillna(method='bfill', inplace=True)

print("\nMissing values after handling:")
print(df.isnull().sum())


# ### Handle Duplicate Data
# Check for duplicate data

# In[13]:


# Cek duplikat data
print("\nJumlah data duplikat:", df.duplicated().sum())

# Hapus data duplikat
df.drop_duplicates(inplace=True)
print("\nJumlah data setelah menghapus duplikat:", len(df))


# ### Sort Data and Set Index
# Ensure the data is sorted chronologically by date and set the 'Date' column as the index.

# In[14]:


df.sort_values('Date', inplace=True)
df.set_index('Date', inplace=True)

print("\nData types after cleaning:")
print(df.dtypes)
print("\nDataset Info:")
df.info()
print("\nFirst 5 rows of processed data:")
print(df.head())


# ## 2. Exploratory Data Analysis (EDA)

# ### Summary Statistics
# Get a statistical overview of the numerical features.

# In[10]:


print("\nSummary Statistics:")
print(df.describe())


# ### Outlier Identification
# Plot the oulier uses boxplot

# In[15]:


# Identify outliers using the IQR method
def identify_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)  # First quartile (25th percentile)
    Q3 = data[column].quantile(0.75)  # Third quartile (75th percentile)
    IQR = Q3 - Q1  # Interquartile range
    lower_bound = Q1 - 1.5 * IQR  # Lower bound
    upper_bound = Q3 + 1.5 * IQR  # Upper bound

    # Identify outliers
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers, lower_bound, upper_bound

# Example: Check for outliers in the 'Close' column
outliers_close, lower_close, upper_close = identify_outliers_iqr(df, 'Close')

print(f"Number of outliers in 'Close': {len(outliers_close)}")
print(f"Lower bound: {lower_close}, Upper bound: {upper_close}")
print("\nOutliers:")
print(outliers_close)


# In[19]:


# Create boxplots to visualize outliers in the price columns
plt.figure(figsize=(14, 10))

# Create subplots for each price column
price_columns = ['Close', 'Open', 'High', 'Low']
for i, column in enumerate(price_columns, 1):
    plt.subplot(2, 2, i)
    sns.boxplot(y=df[column], color='skyblue')
    plt.title(f'Boxplot of {column} Price')
    plt.ylabel('Price (USD)')
    
    # Annotate the number of outliers
    outliers, lower_bound, upper_bound = identify_outliers_iqr(df, column)
    plt.annotate(f'Outliers: {len(outliers)}', 
                 xy=(0.05, 0.95), 
                 xycoords='axes fraction', 
                 fontsize=12,
                 bbox=dict(boxstyle="round", fc="white", alpha=0.8))

plt.tight_layout()
plt.suptitle('Outlier Analysis for Bitcoin Price Data', fontsize=16, y=1.02)
plt.show()

# Visualize outliers in the time series context
plt.figure(figsize=(14, 7))
plt.plot(df.index, df['Close'], label='Close Price', color='blue', alpha=0.7)

# Highlight the outliers
outliers, _, _ = identify_outliers_iqr(df, 'Close')
if not outliers.empty:
    plt.scatter(outliers.index, outliers['Close'], color='red', s=50, label='Outliers')

plt.title('Bitcoin Close Price with Outliers Highlighted')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()


# ### Time Series Visualization
# Plot the key time series: Close price and Volume.

# In[11]:


fig, axes = plt.subplots(3, 1, figsize=(14, 15), sharex=True)

# Plot Close Price
axes[0].plot(df.index, df['Close'], label='Close Price', color='blue')
axes[0].set_title('Bitcoin Close Price Over Time')
axes[0].set_ylabel('Price (USD)')
axes[0].legend()

# Plot Volume
axes[1].bar(df.index, df['Volume'], label='Volume Traded', color='orange', alpha=0.7)
axes[1].set_title('Bitcoin Trading Volume Over Time')
axes[1].set_ylabel('Volume')
axes[1].legend()

# Plot Price Change Percentage
axes[2].plot(df.index, df['Change_Percent'] * 100, label='Daily Change %', color='green', alpha=0.8)
axes[2].set_title('Bitcoin Daily Percentage Change')
axes[2].set_ylabel('Change (%)')
axes[2].axhline(0, color='red', linestyle='--', linewidth=0.8)
axes[2].legend()

plt.xlabel('Date')
plt.tight_layout()
plt.show()


# ### Distribution Analysis
# Look at the distribution of daily returns (percentage change).

# In[12]:


plt.figure(figsize=(10, 6))
sns.histplot(df['Change_Percent'] * 100, bins=50, kde=True)
plt.title('Distribution of Bitcoin Daily Percentage Changes')
plt.xlabel('Daily Change (%)')
plt.ylabel('Frequency')
plt.show()


# ### Volatility Analysis
# Calculate and plot the rolling standard deviation of daily returns (e.g., 30-day volatility).

# In[13]:


df['Daily_Return'] = df['Close'].pct_change()
df['Rolling_Volatility_30D'] = df['Daily_Return'].rolling(window=30).std() * np.sqrt(365) # Annualized

plt.figure(figsize=(14, 7))
df['Rolling_Volatility_30D'].plot(color='purple')
plt.title('Bitcoin 30-Day Rolling Volatility (Annualized)')
plt.ylabel('Annualized Volatility')
plt.xlabel('Date')
plt.show()


# ### EDA Findings
# *   **Price Trend:** The Bitcoin closing price shows significant upward trends over the years, interspersed with periods of high volatility and corrections. There are clear bull and bear market cycles visible.
# *   **Volume:** Trading volume fluctuates considerably. Spikes in volume often coincide with large price movements (both up and down), indicating periods of high market activity and interest.
# *   **Daily Changes:** The distribution of daily percentage changes is centered around zero but has heavy tails (leptokurtic), indicating that extreme price movements (both positive and negative) are more common than in a normal distribution.
# *   **Volatility:** The rolling volatility plot clearly shows periods of high and low volatility (volatility clustering). Volatility tends to spike during market uncertainty or major price swings.
# 

# ## 3. Feature Engineering

# ### Rationale
# We will create additional features from the existing data to potentially improve model performance.
# *   **Lag Features:** Past values (lags) of the target variable ('Close') can be strong predictors for the next value, capturing autocorrelation.
# *   **Rolling Window Features:** Rolling means and standard deviations can capture local trends and volatility patterns over specific periods.
# *   **Date-Based Features:** Day of the week, month, etc., might capture cyclical patterns or seasonality (though strong weekly/monthly seasonality is less common in crypto compared to traditional markets, it's worth checking).

# ### Create Lag Features
# Add lagged 'Close' prices (e.g., 1-day, 7-day lag).

# In[ ]:


n_lags = [1, 3, 7] # Example lag periods
for lag in n_lags:
    df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)


# ### Create Rolling Window Features
# Add rolling mean and standard deviation for 'Close' price and 'Volume'.

# In[ ]:


window_sizes = [7, 30] # Example window sizes (e.g., weekly, monthly)
for window in window_sizes:
    df[f'Close_Rolling_Mean_{window}'] = df['Close'].rolling(window=window).mean()
    df[f'Close_Rolling_Std_{window}'] = df['Close'].rolling(window=window).std()
    df[f'Volume_Rolling_Mean_{window}'] = df['Volume'].rolling(window=window).mean()


# ### Create Date-Based Features
# Extract features like day of week, month, quarter, year.

# In[16]:


df['DayOfWeek'] = df.index.dayofweek # Monday=0, Sunday=6
df['Month'] = df.index.month
df['Quarter'] = df.index.quarter
df['Year'] = df.index.year


# ### Handle NaNs created by lags and rolling windows
# These operations introduce NaNs at the beginning of the dataset.

# In[17]:


print(f"\nData shape before dropping NaNs: {df.shape}")
df.dropna(inplace=True)
print(f"Data shape after dropping NaNs: {df.shape}")

print("\nDataset with engineered features (first 5 rows):")
print(df.head())
print("\nColumns available for modeling:")
print(df.columns.tolist())


# ## 4. Data Preprocessing

# ### Feature Selection
# Select the features to be used as input (X) and the target variable (y).

# In[ ]:


# Target variable
target_col = 'Close'

# Feature columns (include original and engineered features)
feature_cols = [col for col in df.columns if col != target_col and col != 'Change_Percent' and col != 'Daily_Return']

print(f"\nTarget variable: {target_col}")
print(f"Features used for modeling ({len(feature_cols)}): {feature_cols}")

features = df[feature_cols]
target = df[[target_col]]


# ### Data Splitting
# Split the data into training, validation, and testing sets chronologically.

# In[19]:


# Define split proportions
train_size = 0.75
val_size = 0.15
test_size = 0.1


# In[20]:


# Calculate split indices
n = len(df)
train_end_idx = int(n * train_size)
val_end_idx = train_end_idx + int(n * val_size)


# In[21]:


# Split the data
X_train = features[:train_end_idx]
y_train = target[:train_end_idx]

X_val = features[train_end_idx:val_end_idx]
y_val = target[train_end_idx:val_end_idx]

X_test = features[val_end_idx:]
y_test = target[val_end_idx:]

print(f"\nData splitting:")
print(f"Training set shape: X={X_train.shape}, y={y_train.shape}")
print(f"Validation set shape: X={X_val.shape}, y={y_val.shape}")
print(f"Test set shape: X={X_test.shape}, y={y_test.shape}")


# ### Data Scaling (Normalization)
# Scale the features and the target variable using MinMaxScaler to the range [0, 1].
# Fit the scaler ONLY on the training data to prevent data leakage.

# In[22]:


feature_scaler = MinMaxScaler(feature_range=(0, 1))
target_scaler = MinMaxScaler(feature_range=(0, 1))


# In[23]:


# Fit and transform training data
X_train_scaled = feature_scaler.fit_transform(X_train)
y_train_scaled = target_scaler.fit_transform(y_train)


# In[24]:


# Transform validation and test data using the *fitted* scalers
X_val_scaled = feature_scaler.transform(X_val)
y_val_scaled = target_scaler.transform(y_val)

X_test_scaled = feature_scaler.transform(X_test)
y_test_scaled = target_scaler.transform(y_test)

print("\nData scaling completed.")
print(f"X_train_scaled shape: {X_train_scaled.shape}")
print(f"y_train_scaled shape: {y_train_scaled.shape}")


# ### Create Time Series Sequences
# Create sequences of data (e.g., use the last 60 days of data to predict the next day).

# In[ ]:


def create_sequences(X_data, y_data, sequence_length):
    """
    Generates sequences of data for time series forecasting.
    Args:
        X_data (np.array): Scaled feature data.
        y_data (np.array): Scaled target data.
        sequence_length (int): Number of time steps in each input sequence.
    Returns:
        tuple: (X_sequence, y_sequence) numpy arrays.
    """
    X_sequence, y_sequence = [], []
    for i in range(len(X_data) - sequence_length):
        X_sequence.append(X_data[i:(i + sequence_length)])
        y_sequence.append(y_data[i + sequence_length])
    return np.array(X_sequence), np.array(y_sequence)


# In[ ]:


# Define sequence length (number of past time steps to use for prediction)
SEQUENCE_LENGTH = 60

# Create sequences for train, validation, and test sets
X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, SEQUENCE_LENGTH)
X_val_seq, y_val_seq = create_sequences(X_val_scaled, y_val_scaled, SEQUENCE_LENGTH)
X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_scaled, SEQUENCE_LENGTH)

print("\nSequence creation completed.")
print(f"X_train_seq shape: {X_train_seq.shape}")
print(f"y_train_seq shape: {y_train_seq.shape}")
print(f"X_val_seq shape: {X_val_seq.shape}")
print(f"y_val_seq shape: {y_val_seq.shape}")
print(f"X_test_seq shape: {X_test_seq.shape}")
print(f"y_test_seq shape: {y_test_seq.shape}")


# In[27]:


# Get the number of features after creating sequences
num_features = X_train_seq.shape[2]
print(f"\nNumber of features per time step: {num_features}")


# ## 5. Modeling

# ### Model Architecture Setup
# Define hyperparameters and input shape.

# In[28]:


INPUT_SHAPE = (SEQUENCE_LENGTH, num_features)
LSTM_UNITS = 64
CNN_FILTERS = 64
KERNEL_SIZE = 3
DROPOUT_RATE = 0.2
EPOCHS = 30
BATCH_SIZE = 32


# In[29]:


# Callbacks for training
early_stopping = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
# Save the best model checkpoint
model_checkpoint = ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True)
# Reduce learning rate on plateau
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.133, patience=3, min_lr=1e-6)


# ### Model 1: LSTM

# In[30]:


def build_lstm_model(input_shape, lstm_units, dropout_rate):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(lstm_units, return_sequences=True), # return_sequences=True if stacking LSTMs
        Dropout(dropout_rate),
        LSTM(lstm_units // 2, return_sequences=False), # Last LSTM layer returns only the final output
        Dropout(dropout_rate),
        Dense(32, activation='mish'),
        Dense(1) # Output layer: 1 neuron for predicting the single 'Close' price
    ])
    model.compile(optimizer=AdamW(learning_rate=1e-4), loss='mse') # Mean Squared Error for regression
    return model

lstm_model = build_lstm_model(INPUT_SHAPE, LSTM_UNITS, DROPOUT_RATE)
lstm_model.summary()


# #### Train LSTM Model

# In[31]:


print("\nTraining LSTM model...")
history_lstm = lstm_model.fit(
    X_train_seq, y_train_seq,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_val_seq, y_val_seq),
    callbacks=[early_stopping, reduce_lr],
    verbose=1 # Set to 0 for less output, 1 for progress bar
)
print("LSTM model training finished.")


# #### Plot LSTM Training History
# 

# In[32]:


def plot_training_history(history, model_name):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} - Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.show()

plot_training_history(history_lstm, "LSTM")


# ### Model 2: CNN-LSTM Hybrid

# In[33]:


def build_cnn_lstm_model(input_shape, cnn_filters, kernel_size, lstm_units, dropout_rate):
    model = Sequential([
        Input(shape=input_shape),
        # CNN layers to extract features from sequences
        Conv1D(filters=cnn_filters, kernel_size=kernel_size, activation='mish', padding='same'),
        # MaxPooling1D(pool_size=2), # Optional: Downsample
        Conv1D(filters=cnn_filters // 2, kernel_size=kernel_size, activation='mish', padding='same'),
        MaxPooling1D(pool_size=2),
        # LSTM layers to process sequences of features extracted by CNN
        LSTM(lstm_units, return_sequences=True),
        Dropout(dropout_rate),
        LSTM(lstm_units // 2, return_sequences=False),
        Dropout(dropout_rate),
        Dense(32, activation='mish'),
        Dense(1) # Output layer
    ])
    model.compile(optimizer=AdamW(learning_rate=1e-4), loss='mse')
    return model

cnn_lstm_model = build_cnn_lstm_model(INPUT_SHAPE, CNN_FILTERS, KERNEL_SIZE, LSTM_UNITS, DROPOUT_RATE)
cnn_lstm_model.summary()


# #### Train CNN-LSTM Model

# In[34]:


print("\nTraining CNN-LSTM model...")
history_cnn_lstm = cnn_lstm_model.fit(
    X_train_seq, y_train_seq,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_val_seq, y_val_seq),
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)
print("CNN-LSTM model training finished.")


# #### Plot CNN-LSTM Training History

# In[35]:


plot_training_history(history_cnn_lstm, "CNN-LSTM")


# ## 6. Evaluation
# 
# 

# ### Make Predictions on Test Set
# Use the trained models to predict on the unseen test set. Remember predictions are scaled.

# In[36]:


y_pred_lstm_scaled = lstm_model.predict(X_test_seq)
y_pred_cnn_lstm_scaled = cnn_lstm_model.predict(X_test_seq)


# ### Inverse Transform Predictions
# Convert the scaled predictions and actual values back to the original price scale.

# In[ ]:


y_pred_lstm = target_scaler.inverse_transform(y_pred_lstm_scaled)
y_pred_cnn_lstm = target_scaler.inverse_transform(y_pred_cnn_lstm_scaled)

# The actual y values start from SEQUENCE_LENGTH steps into the original test set
y_test_actual = target[val_end_idx + SEQUENCE_LENGTH:].values


# In[38]:


# Adjust y_test_actual shape if needed (e.g., due to sequence creation)
if len(y_test_actual) != len(y_pred_lstm):
     print(f"Adjusting y_test_actual length from {len(y_test_actual)} to match predictions {len(y_pred_lstm)}")
     y_test_actual = y_test[SEQUENCE_LENGTH:].values


# ### Calculate Evaluation Metrics
# Use MAE, RMSE, and MAPE to evaluate model performance.

# In[39]:


def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100 # Percentage
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}


# In[40]:


metrics_lstm = calculate_metrics(y_test_actual, y_pred_lstm)
metrics_cnn_lstm = calculate_metrics(y_test_actual, y_pred_cnn_lstm)

print("\nEvaluation Metrics on Test Set:")
print(f"LSTM Model: {metrics_lstm}")
print(f"CNN-LSTM Model: {metrics_cnn_lstm}")


# ### Compare Model Performance

# In[41]:


metrics_df = pd.DataFrame([metrics_lstm, metrics_cnn_lstm], index=['LSTM', 'CNN-LSTM'])
print("\nModel Comparison:")
print(metrics_df)


# ### Visualize Actual vs. Predicted Prices

# In[42]:


# Get the dates corresponding to the test predictions
test_dates = df.index[val_end_idx + SEQUENCE_LENGTH:]

plt.figure(figsize=(14, 7))
plt.plot(test_dates, y_test_actual, label='Actual Price', color='blue', linewidth=2)
plt.plot(test_dates, y_pred_lstm, label='LSTM Predicted Price', color='red', linestyle='--', alpha=0.8)
plt.plot(test_dates, y_pred_cnn_lstm, label='CNN-LSTM Predicted Price', color='green', linestyle='--', alpha=0.8)

plt.title('Bitcoin Price Prediction vs Actual (Test Set)')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()


# In[ ]:


# Zoom in on a smaller period for better visibility if needed
zoom_start = test_dates[0]
zoom_end = test_dates[min(100, len(test_dates)-1)]

plt.figure(figsize=(14, 7))
plt.plot(test_dates, y_test_actual, label='Actual Price', color='blue', linewidth=2)
plt.plot(test_dates, y_pred_lstm, label='LSTM Predicted Price', color='red', linestyle='--', alpha=0.8)
plt.plot(test_dates, y_pred_cnn_lstm, label='CNN-LSTM Predicted Price', color='green', linestyle='--', alpha=0.8)

plt.title('Bitcoin Price Prediction vs Actual (Test Set - Zoomed In)')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.xlim(zoom_start, zoom_end)
plt.legend()
plt.show()


# ## 7. Testing / Future Simulation (Next 30 Days)

# ### Select Best Model
# Choose the model with better performance on the test set (e.g., lower RMSE or MAE).

# In[44]:


best_model = lstm_model if metrics_lstm['RMSE'] <= metrics_cnn_lstm['RMSE'] else cnn_lstm_model
best_model_name = "LSTM" if best_model == lstm_model else "CNN-LSTM"
print(f"\nSelected best performing model: {best_model_name}")


# ### Prepare Input for Simulation
# Get the last sequence from the *entire* available dataset (including test data) to start the prediction.

# In[45]:


# Combine all scaled features data
all_features_scaled = np.concatenate((X_train_scaled, X_val_scaled, X_test_scaled), axis=0)

# Get the last known sequence
last_sequence = all_features_scaled[-SEQUENCE_LENGTH:]
# Reshape it for model input: (1, sequence_length, num_features)
current_sequence = last_sequence.reshape((1, SEQUENCE_LENGTH, num_features))


# ### Simulate Future Predictions Iteratively
# 

# In[ ]:


forecast_horizon = 30 # Predict the next 30 days
future_predictions_scaled = []

print(f"\nSimulating forecast for the next {forecast_horizon} days using {best_model_name}...")

for i in range(forecast_horizon):
    # Predict the next time step (scaled)
    next_pred_scaled = best_model.predict(current_sequence)[0, 0]

    # Store the scaled prediction
    future_predictions_scaled.append(next_pred_scaled)

    # Create the feature vector for the predicted step
    next_step_features = current_sequence[0, -1, :].copy()
    new_step_features_scaled = current_sequence[0, -1, :].copy()
    next_feature_vector = current_sequence[0, -1, :].copy()
    next_feature_vector[0] = next_pred_scaled

    # Append this *approximated* feature vector for the next step
    new_sequence_step = next_feature_vector.reshape(1, 1, num_features)

    # Roll the sequence: remove the oldest step, append the new approximated step
    current_sequence = np.append(current_sequence[:, 1:, :], new_sequence_step, axis=1)


print("Simulation finished.")


# ### Inverse Transform Forecast
# Convert the scaled forecast back to the original price scale.

# In[47]:


future_predictions = target_scaler.inverse_transform(np.array(future_predictions_scaled).reshape(-1, 1))


# ### Create Future Dates Index
# Generate dates for the forecast period.

# In[48]:


last_date = df.index[-1]
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_horizon, freq='D')


# ### Visualize Forecast

# In[49]:


plt.figure(figsize=(14, 7))

# Plot historical data (last N days)
history_days = 90
plt.plot(df.index[-history_days:], df['Close'][-history_days:], label='Historical Close Price', color='blue')

# Plot the forecast
plt.plot(future_dates, future_predictions, label='Forecasted Price', color='red', linestyle='--')

plt.title(f'Bitcoin Price Forecast for Next {forecast_horizon} Days ({best_model_name})')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()


# ### Simulation Explanation
# The plot above shows the historical closing prices for the last 90 days and the simulated forecast for the next 30 days using the best-performing model CNN-LSTM (Hybrid).
# 
# **Important Considerations & Limitations:**
# *   **Iterative Prediction:** The forecast is generated iteratively. The prediction for day 1 is used as part of the input to predict day 2, and so on. This means errors can accumulate over the forecast horizon.
# *   **Feature Approximation:** A major challenge in multi-step forecasting with multivariate inputs (like Open, High, Low, Volume, etc.) using a single-output model (predicting only 'Close') is determining the future values of the input features. In this simulation, we made a strong simplifying assumption: the feature vector for the next predicted step was approximated based on the last known step, with only a primary price feature updated using the prediction. This is not realistic, as other features (Volume, High, Low) also change dynamically. A more robust approach would involve:
#     *   Training a model using only lagged values of the target variable ('Close').
#     *   Building a multi-output model that predicts all necessary features for the next time step.
#     *   Using external forecasts or assumptions for the future input features.
# *   **Model Confidence:** The accuracy of the forecast typically decreases as the forecast horizon increases. Deep learning models capture past patterns but cannot predict unforeseen future events (e.g., sudden market crashes, regulatory changes, major news).
# 
# Therefore, this 30-day forecast should be interpreted with caution, serving more as an illustration of the model's potential trend extrapolation based on learned patterns, rather than a guaranteed future outcome.

# ## 8. Conclusion
# 
# ### Summary of Results
# In this project, we developed and compared LSTM and CNN-LSTM hybrid models for predicting daily Bitcoin closing prices. Both models were trained on historical data from 2016 onwards, incorporating feature engineering techniques like lags and rolling statistics.
# *   The CNN-LSTM hybrid model achieved slightly better performance on the test set, with an RMSE of 10810,2 and a MAPE of 10,13%.
# *   Visualizations showed that both models could capture the general trend of the Bitcoin price on the test set, although predicting the exact magnitude and timing of fluctuations remains challenging.
# *   The future simulation provided a potential price trajectory for the next 30 days, highlighting the model's predictive capability but also the inherent uncertainties and limitations of long-term forecasting, especially with the simplified feature handling during simulation.
# 
# ### Insights
# *   **Feature Engineering:** Adding lag features and rolling statistics seemed beneficial, providing the models with more context about recent trends and volatility.
# *   **Model Complexity:** The CNN-LSTM model, while potentially capable of extracting more complex patterns, did not significantly outperform the standard LSTM in this specific setup. This could be due to the nature of the data, the chosen hyperparameters, or the architecture configuration. Further tuning might be required.
# *   **Volatility:** EDA confirmed Bitcoin's high volatility, making precise point predictions difficult. Models are generally better at capturing trends than exact price levels day-to-day.
# 
# ### Suggestions for Future Work
# *   **Hyperparameter Tuning:** Systematically tune hyperparameters (e.g., sequence length, LSTM units, CNN filters, learning rate, dropout rate) using techniques like Grid Search, Random Search, or Bayesian Optimization.
# *   **Advanced Architectures:** Explore other deep learning architectures like Attention mechanisms or Transformers, which have shown promise in sequence modeling.
# *   **Multivariate Forecasting:** Implement models that predict multiple outputs (e.g., Close, High, Low, Volume) simultaneously, which could lead to more consistent future simulations.
# *   **Exogenous Variables:** Incorporate external factors like market sentiment (from social media or news), blockchain metrics (e.g., hash rate, transaction count), or macroeconomic indicators.
# *   **Ensemble Methods:** Combine predictions from multiple models (e.g., LSTM, CNN-LSTM, ARIMA, Prophet) to potentially improve robustness and accuracy.
# *   **Refined Simulation:** Develop a more sophisticated simulation strategy that forecasts or models the evolution of all input features, not just the target variable.
# *   **Longer Evaluation:** Evaluate performance over multiple rolling forecast origins to get a more robust measure of generalization.
