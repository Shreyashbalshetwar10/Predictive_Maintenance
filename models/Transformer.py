# Transformer for Complete Dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
import joblib
import os
import random

def reset_seeds(seed_value= 81):
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)

    np.random.seed(seed_value)

    tf.random.set_seed(seed_value)
    print(f"Random seeds set to {seed_value}")

reset_seeds()

sensor_columns = [f'sensor_{i}' for i in range(1, 22)]

# Data Loading
FD001 = pd.read_csv(r'data/train_FD001.txt', sep=" ", header=None)
FD001.dropna(axis=1, how='all', inplace=True)
FD001.columns = ['engine_id', 'cycle'] + ['S_1', 'S_2', 'S_3'] + sensor_columns

FD002 = pd.read_csv(r'data/train_FD002.txt', sep=" ", header=None)
FD002.dropna(axis=1, how='all', inplace=True)
FD002.columns = ['engine_id', 'cycle'] + ['S_1', 'S_2', 'S_3'] + sensor_columns

FD003 = pd.read_csv(r'data/train_FD003.txt', sep=" ", header=None)
FD003.dropna(axis=1, how='all', inplace=True)
FD003.columns = ['engine_id', 'cycle'] + ['S_1', 'S_2', 'S_3'] + sensor_columns

FD004 = pd.read_csv(r'data/train_FD004.txt', sep=" ", header=None)
FD004.dropna(axis=1, how='all', inplace=True) 
FD004.columns = ['engine_id', 'cycle'] + ['S_1', 'S_2', 'S_3'] + sensor_columns

FD002['engine_id'] += FD001['engine_id'].max()
FD003['engine_id'] += FD002['engine_id'].max()
FD004['engine_id'] += FD003['engine_id'].max()

data = pd.concat([FD001, FD002, FD003, FD004], ignore_index=True)

input = ['S_1', 'S_2', 'S_3'] + sensor_columns

all_engine_data = []
for engine_id in data['engine_id'].unique():
    engine_data = data[data['engine_id'] == engine_id]
    df_engine = engine_data.reset_index(drop=True)
    # Apply Rolling Mean (Window of 5 cycles)
    features = df_engine[input].rolling(window=5).mean().fillna(method='bfill').values
    
    all_engine_data.append(features)  # Exclude engine ID column

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

# Positional Embedding Layer
layers = tf.keras.layers
class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.position_embeddings = layers.Embedding(input_dim=sequence_length, output_dim=output_dim)
        self.projection = layers.Dense(output_dim)

    def call(self, inputs):
        length = tf.shape(inputs)[1]
        positions = tf.range(start=0, limit=length, delta=1)

        embedded_positions = self.position_embeddings(positions)

        projected_inputs = self.projection(inputs)

        return projected_inputs + embedded_positions
    
class OrderPositionalEncoding(layers.Layer):
    def __init__(self, sequence_length, output_dim, **kwargs):
        super(OrderPositionalEncoding, self).__init__(**kwargs)

        self.pe = self.get_position_encoding(sequence_length, output_dim)

    def get_position_encoding(self, seq_len, d_model):
        position = np.arange(seq_len)[:, np.newaxis]

        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

        pe = np.zeros((seq_len, d_model))

        pe[:, 0::2] = np.sin(position * div_term)

        pe[:, 1::2] = np.cos(position * div_term)

        pe = pe[np.newaxis, :, :]
        return tf.constant(pe, dtype=tf.float32)

    def call(self, inputs, *args, **kwargs):

        if isinstance(inputs, tf.SparseTensor):
            inputs = tf.sparse.to_dense(inputs)

        return inputs + tf.cast(self.pe, dtype=inputs.dtype)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super(OrderPositionalEncoding, self).get_config()
        config.update({
            "sequence_length": self.sequence_length,
            "output_dim": self.output_dim,
        })
        return config
    
class WarmupScheduler(tf.keras.callbacks.Callback):
    def __init__(self, warmup_steps=1000, max_lr=1e-3):
        super(WarmupScheduler, self).__init__()
        self.warmup_steps = warmup_steps
        self.max_lr = max_lr

    def on_train_batch_begin(self, batch, logs=None):

        step = self.model.optimizer.iterations
        step_float = tf.cast(step, tf.float32)

        lr = self.max_lr * (step_float / self.warmup_steps)

        lr = tf.minimum(lr, self.max_lr)

        self.model.optimizer.learning_rate.assign(lr)
    
# The Custom Loss Function (Negative Log Likelihood)
# It predicts both the mean (RUL) and the log variance (uncertainty).
def negative_log_likelihood(y_true, y_pred):

    y_true = tf.cast(y_true, tf.float32)
    mu = y_pred[:, 0:1]
    log_var = y_pred[:, 1:2]

    var = tf.exp(log_var)
    
    loss = 0.5 * (log_var + tf.square(y_true - mu) / var)
    return tf.reduce_mean(loss)

# For Huber + NLL
class ProbabilisticHuberLoss(tf.keras.losses.Loss):
    def __init__(self, delta=0.1, name="prob_huber_loss"):
        super().__init__(name=name)
        self.delta = delta

    def call(self, y_true, y_pred):
        pred_rul = y_pred[:, 0:1]     
        log_variance = y_pred[:, 1:2] 

        log_variance = tf.maximum(log_variance, -10.0)

        error = y_true - pred_rul
        abs_error = tf.abs(error)

        quadratic = tf.minimum(abs_error, self.delta)
        linear = abs_error - quadratic
        huber_loss = 0.5 * quadratic**2 + self.delta * linear

        loss = huber_loss * tf.exp(-log_variance) + 0.5 * log_variance
        
        return tf.reduce_mean(loss)

def mse_only_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    mu = y_pred[:, 0:1]
    return tf.reduce_mean(tf.square(y_true - mu))

def rmse_metric(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    mu = y_pred[:, 0:1]
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - mu)))

# Train-test split + scaling
X_train, X_test, y_train, y_test = train_test_split(all_engine_data, ruls, test_size=0.2, random_state=42)

X_train_stacked = np.vstack(X_train)
y_train_stacked = np.hstack(y_train)

scaler = StandardScaler()

scaler.fit(X_train_stacked)

X_train_scaled = [scaler.transform(engine_data) for engine_data in X_train]
X_test_scaled = [scaler.transform(engine_data) for engine_data in X_test]

# Scale RUL values 
# Added because the model was not learning the data but rather getting results by exploding the rul values
scaler_y = MinMaxScaler(feature_range=(0, 1))
scaler_y.fit(y_train_stacked.reshape(-1, 1))
y_train_scaled = [scaler_y.transform(np.array(rul).reshape(-1, 1)).flatten() for rul in y_train]
y_test_scaled = [scaler_y.transform(np.array(rul).reshape(-1, 1)).flatten() for rul in y_test]

# Sliding window function
def create_sliding_windows(X, y, window_size=50):
    X_windows = []
    y_windows = []
    for engine_data, rul_data in zip(X, y):
        for i in range(len(engine_data) - window_size + 1):
            X_windows.append(engine_data[i:i+window_size])
            y_windows.append(rul_data[i+window_size-1])
    return np.array(X_windows), np.array(y_windows)

X_train_windows, y_train_windows = create_sliding_windows(X_train_scaled, y_train_scaled)
X_test_windows, y_test_windows = create_sliding_windows(X_test_scaled, y_test_scaled)


# Data Augmentation
class TimeSeriesAugmentation(layers.Layer):
    def __init__(self, noise_level=0.01, scale_range=0.02, **kwargs):
        super().__init__(**kwargs)
        self.noise_level = noise_level 
        self.scale_range = scale_range 

    def call(self, inputs, training=None, *args, **kwargs):
        if not training:
            return inputs
        
        noise = tf.random.normal(shape=tf.shape(inputs), mean=0.0, stddev=self.noise_level)
        augmented = inputs + noise

        batch_size = tf.shape(inputs)[0]
        scales = tf.random.uniform(shape=(batch_size, 1, 1), minval=1.0 - self.scale_range, maxval=1.0 + self.scale_range)
        augmented = augmented * scales
        
        return augmented
    
    def get_config(self):
        config = super().get_config()
        config.update({"noise_level": self.noise_level})
        return config

    
# Model definition
models = tf.keras.models

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Attention Layer
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    
    # Multi-Head Attention
    x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    
    res = x + inputs

    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)

    return x + res

def build_probabilistic_transformer(input_shape):
    inputs = layers.Input(shape=input_shape)

    # Add Data Augmentation Layer
    x = TimeSeriesAugmentation(noise_level=0.01, scale_range=0.02)(inputs, training=True)
    d_model = 64
    x = layers.Dense(d_model)(x)

    x = x * tf.math.sqrt(tf.cast(d_model, tf.float32))
    
    # Positional Embedding
    x = OrderPositionalEncoding(sequence_length=50, output_dim=d_model)(x)
    
    # Transformer Blocks
    for _ in range(2):
        x = transformer_encoder(x, head_size=64, num_heads=4, ff_dim=64, dropout=0.3)

    x = layers.Flatten()(x)
    
    x = layers.Dense(64)(x)
    x = layers.LeakyReLU(alpha=0.01)(x)
    x = layers.Dropout(0.2)(x)
    
    outputs = layers.Dense(1, activation='linear')(x)

    model = models.Model(inputs, outputs)

    loss_fn = tf.keras.losses.Huber(delta=0.1)
    
    model.compile(optimizer='adam', loss=loss_fn, metrics=[rmse_metric])
    return model

loss_fn = tf.keras.losses.Huber(delta=0.1)

def build_two_stage_model(input_shape):
    inputs = layers.Input(shape=input_shape)

    # Data Augmentation Layer
    x = TimeSeriesAugmentation(noise_level=0.01, scale_range=0.02)(inputs, training=True)
    
    d_model = 64
    x = layers.Dense(d_model)(x)
    
    x = x * tf.math.sqrt(tf.cast(d_model, tf.float32))
    
    # Positional Embedding
    x = OrderPositionalEncoding(sequence_length=50, output_dim=d_model)(x)
    
    # Transformer Blocks
    for _ in range(2):
        x = transformer_encoder(x, head_size=64, num_heads=4, ff_dim=64, dropout=0.3)

    x = layers.GlobalAveragePooling1D()(x)
    
    x = layers.Dense(64)(x)
    x = layers.LeakyReLU(alpha=0.01)(x)
    x = layers.Dropout(0.2)(x)
    
    features = layers.Dense(128, activation='relu')(x)
    
    # --- HEAD 1: MEAN (The Trend) ---
    mean_out = layers.Dense(1, activation='linear', name='mean')(features)
    
    # --- HEAD 2: UNCERTAINTY (The Variance) ---
    var_out = layers.Dense(1, activation='softplus', bias_initializer=tf.keras.initializers.Constant([-4.0]),name='var')(features)
    
    return models.Model(inputs=inputs, outputs=[mean_out, var_out])

input_shape = (50, X_train_windows.shape[2])

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=15,          
    restore_best_weights=True,
    mode='min'
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.5,          
    patience=7,            
    min_lr=1e-6,          
    verbose=1
)

BATCH_SIZE = 16
EPOCHS = 100
MAX_LR = 5e-5

total_steps = (len(X_train_windows) // BATCH_SIZE) * EPOCHS

print(f"Total Decay Steps: {total_steps}")

warmup_steps = int(0.1 * total_steps)

print(f"Total Steps: {total_steps}")
print(f"Warmup Steps: {warmup_steps}")

warmup_callback = WarmupScheduler(
    warmup_steps=warmup_steps, 
    max_lr=MAX_LR
)

lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=0.0,      # Start at 0
    decay_steps=total_steps,        # The total length of training
    warmup_target=MAX_LR,             # Peak LR
    warmup_steps=warmup_steps       # Steps to get to Peak
)

optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, clipvalue=0.5)

class PrintLR(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.learning_rate
        # Handle schedule objects
        if isinstance(lr, tf.keras.optimizers.schedules.LearningRateSchedule):
            current_step = self.model.optimizer.iterations
            lr = lr(current_step)
        print(f"\n📢 Epoch {epoch+1} LR: {float(lr):.7f}")

def build_cnn_transformer(input_shape):
    inputs = layers.Input(shape=input_shape)
    
    x = layers.Conv1D(filters=64, kernel_size=3, padding="same", activation="relu")(inputs)
    x = layers.Conv1D(filters=64, kernel_size=3, padding="same", activation="relu")(x)
    # Pool to reduce noise
    x = layers.MaxPooling1D(pool_size=2)(x) 

    x = transformer_encoder(x, head_size=64, num_heads=4, ff_dim=64, dropout=0.1)

    x = layers.GlobalAveragePooling1D()(x)

    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(1)(x)

    model = models.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='mse')
    return model

model = build_two_stage_model(input_shape=(50, 24))

losses_stage1 = {
    "mean": tf.keras.losses.Huber(delta=0.1),
    "var": None                                
}

loss_weights_stage1 = {"mean": 1.0, "var": 0.0}

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss=losses_stage1,
              loss_weights=loss_weights_stage1,
              metrics={'mean': tf.keras.metrics.RootMeanSquaredError(name='rmse')})

# Casting for matching types
print("--- FIXING DATA TYPES ---")

X_train_windows = np.array(X_train_windows).astype(np.float32)
X_test_windows  = np.array(X_test_windows).astype(np.float32)

y_train_windows = y_train_windows.astype(np.float32)
y_test_windows = y_test_windows.astype(np.float32)

print(f"X_train shape: {X_train_windows.shape} | Type: {X_train_windows.dtype}")
print(f"y_train shape: {y_train_windows.shape} | Type: {y_train_windows.dtype}")

if np.isnan(X_train_windows).any():
    print("WARNING: NaNs found in X_train! Replacing with 0.0")
    X_train_windows = np.nan_to_num(X_train_windows)

# Train Stage 1
print("\n=== STAGE 1: Training for Accuracy (Huber) ===")
history1 = model.fit(
    X_train_windows, 
    {"mean": y_train_windows, "var": y_train_windows},
    validation_split=0.1,
    epochs=50, 
    batch_size=16
)

print("\n=== STAGE 2: Training for Uncertainty ===")

for layer in model.layers:
    layer.trainable = False

model.get_layer("var").trainable = True

preds_stage1 = model.predict(X_train_windows)[0].flatten()
residuals_sq = (y_train_windows - preds_stage1) ** 2

preds_test_stage1 = model.predict(X_test_windows)[0].flatten()
residuals_sq_test = (y_test_windows - preds_test_stage1) ** 2

losses_stage2 = {
    "mean": None,
    "var": "mse" 
}
loss_weights_stage2 = {"mean": 0.0, "var": 1.0}

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss=losses_stage2,
              loss_weights=loss_weights_stage2)

#Train Stage 2
history2 = model.fit(
    X_train_windows, 
    {"mean": y_train_windows, "var": residuals_sq},
    validation_split=0.1,
    epochs=20, 
    batch_size=16
)

# Predict and evaluate
y_pred = model.predict(X_test_windows)
y_pred[0] = scaler_y.inverse_transform(y_pred[0].reshape(-1, 1)).flatten()

y_test_windows_1 = scaler_y.inverse_transform(y_test_windows.reshape(-1, 1)).flatten()
mse = mean_squared_error(y_test_windows_1, y_pred[0])

rmse = np.sqrt(mse)
uncertainty = np.sqrt(y_pred[1]).flatten() * scaler_y.data_range_[0]
print(f"Test RMSE: {rmse:.4f} | Uncertainty: {uncertainty.mean():.4f}")

y_pred_train = model.predict(X_train_windows)
y_pred_train[0] = scaler_y.inverse_transform(y_pred_train[0].reshape(-1, 1)).flatten()  

y_train_windows_2 = scaler_y.inverse_transform(y_train_windows.reshape(-1, 1)).flatten()
mse_train = mean_squared_error(y_train_windows_2, y_pred_train[0])

rmse_train = np.sqrt(mse_train)
uncertainty_train = np.sqrt(y_pred_train[1]).flatten() * scaler_y.data_range_[0]  
print(f"Train RMSE: {rmse_train:.4f} | Uncertainty: {uncertainty_train.mean():.4f}")

# NASA Score Calculation
def nasa_score(y_true, y_pred):
    y_pred = y_pred[0]
    
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    d = y_pred - y_true

    scores = np.where(
        d < 0,
        np.exp(-d / 13) - 1,  
        np.exp(d / 10) - 1     
    )
    
    return np.sum(scores)

my_score = nasa_score(y_test_windows_1, y_pred)

print(f"NASA Safety Score: {my_score:.2f}")


# Save the model
model.save('Transformer_Uncertainty_model.h5')
print("Model saved as 'Transformer_Uncertainty_model.h5'.")

# Save the scaler
joblib.dump(scaler, 'U_T.gz')
joblib.dump(scaler_y, 'U_T_y.gz')
print("Scalers saved successfully.")


def nasa_score_vectorized(y_true, y_pred):
    d = y_pred - y_true
    scores = np.where(
        d < 0,
        np.exp(-d / 13) - 1,
        np.exp(d / 10) - 1
    )
    return np.sum(scores)

safety_factor = 1.5 

y_pred_safe = y_pred[0] - (safety_factor * uncertainty)
y_pred_safe = np.maximum(y_pred_safe, 0)

y_pred_safe = np.maximum(y_pred_safe, 0)
y_pred_safe = np.minimum(y_pred_safe, 150.0)

safe_mse = mean_squared_error(y_test_windows_1, y_pred_safe)
safe_rmse = np.sqrt(safe_mse)

safe_score = nasa_score_vectorized(y_test_windows_1, y_pred_safe)

print(f"--- RESULTS WITH SAFETY FACTOR {safety_factor} ---")
print(f"Original RMSE: {rmse:.4f}")
print(f"Safe RUL RMSE: {safe_rmse:.4f}")
print(f"Safe RUL NASA Score: {safe_score:,.0f}")

subset_n = 100
preds = model.predict(X_test_windows[:subset_n])
preds_processed = scaler_y.inverse_transform(preds[0].reshape(-1, 1))  # Mean (RUL)
pred_rul = np.array(preds_processed).flatten()
pred_rul_safe = pred_rul - (safety_factor * np.sqrt(preds[1]).flatten() * scaler_y.data_range_[0])

true_rul = np.array(y_test_windows_1[:subset_n]).flatten()

plt.figure(figsize=(12, 5))
plt.plot(true_rul, label='Actual Truth', color='blue', alpha=0.6)
plt.plot(pred_rul_safe, label='Model Prediction', color='red', linewidth=3)
plt.title("Adjusted for Safety Factor")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('Safe_rul.png')
print("Visual result saved as 'Safe_rul.png'.")
