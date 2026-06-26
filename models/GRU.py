import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf

def reset_seeds(seed_value= 81):
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)

    np.random.seed(seed_value)

    tf.random.set_seed(seed_value)
    print(f"Random seeds set to {seed_value}")

reset_seeds()

data_1 = np.loadtxt(r'/home/shreyash/Projects/Predictive_Maintenance/CMaps/train_FD001.txt')

all_engine_data = []
for engine_id in range(1, 101):
    mask = (data_1[:, 0] == engine_id)
    engine_data = data_1[mask]
    df_engine = pd.DataFrame(engine_data)
    # Apply Rolling Mean (Window of 5 cycles)
    features = df_engine.rolling(window=5).mean().fillna(method='bfill').values
    
    all_engine_data.append(features[:, 1:])  # Exclude engine ID column

ruls = []
for i in range(len(all_engine_data)):
    rul = []
    for j in range(len(all_engine_data[i])):
        y_i = len(all_engine_data[i]) - j - 1
        rul.append(y_i)
    ruls.append(rul)

# Cap RUL at 125
for i in range(len(ruls)):
    for j in range(len(ruls[i])):
        if ruls[i][j] > 125:
            ruls[i][j] = 125

X_train, X_test, y_train, y_test = train_test_split(all_engine_data, ruls, test_size=0.2, random_state=42)

# Normalize the data
scaler = MinMaxScaler()
x_train_flat = np.vstack(X_train)
scaler.fit(x_train_flat)
X_train_scaled = [scaler.transform(engine_data) for engine_data in X_train]
X_test_scaled = [scaler.transform(engine_data) for engine_data in X_test]

# Model definition
GRU = tf.keras.layers.GRU
Dense = tf.keras.layers.Dense
Dropout = tf.keras.layers.Dropout
Sequential = tf.keras.models.Sequential

model = Sequential()
model.add(GRU(100, activation='tanh', return_sequences=True, input_shape=(50, X_train_scaled[0].shape[1])))
model.add(Dropout(0.2))
model.add(GRU(50, activation='tanh'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='relu'))

# Sliding window function
def create_sliding_windows(X, y, window_size=50):
    X_windows = []
    y_windows = []
    for engine_data, rul_data in zip(X, y):
        for i in range(len(engine_data) - window_size + 1):
            X_windows.append(engine_data[i:i+window_size])
            y_windows.append(rul_data[i+window_size-1])  # RUL at the end of the window
    return np.array(X_windows), np.array(y_windows)

X_train_windows, y_train_windows = create_sliding_windows(X_train_scaled, y_train)
X_test_windows, y_test_windows = create_sliding_windows(X_test_scaled, y_test)

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['root_mean_squared_error'])

# Train the model
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.5, 
    patience=7, 
    #min_lr=0.00001, 
    verbose=1
)

history = model.fit(X_train_windows, y_train_windows, epochs=50, batch_size=32, validation_split=0.1, callbacks=[early_stop, reduce_lr], verbose=1)

# Predict and evaluate
y_pred = model.predict(X_test_windows)
mse = mean_squared_error(y_test_windows, y_pred)
rmse = np.sqrt(mse)
print(f"GRU Test RMSE: {rmse}")

y_pred_2 = model.predict(X_train_windows)
mse_2 = mean_squared_error(y_train_windows, y_pred_2)
rmse_2 = np.sqrt(mse_2)
print(f"GRU Train RMSE: {rmse_2}")

# Plot training & validation loss values
sorted_indices = np.argsort(y_test_windows)
y_test_sorted = y_test_windows[sorted_indices]
y_pred_sorted = y_pred[sorted_indices]

plt.figure(figsize=(10, 6))
plt.plot(y_test_sorted, label='Actual RUL', color='black', linewidth=2)
plt.plot(y_pred_sorted, label='Predicted RUL', color='red', alpha=0.7, linewidth=1)

plt.title('Prediction Accuracy: Actual vs Predicted')
plt.xlabel('Test Samples (Sorted by Life)')
plt.ylabel('RUL (Cycles)')
plt.legend()
plt.grid(True)
plt.savefig('gru_result.png')
print("Plot saved as gru_result.png")
