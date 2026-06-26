import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

data_1 = np.loadtxt(r'data/train_FD002.txt')

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

X_train, X_test, y_train, y_test = train_test_split(all_engine_data, ruls, test_size=0.2, random_state=42)

X_train_flat = np.vstack(X_train)
y_train_flat = np.hstack(y_train)

X_test_flat = np.vstack(X_test)
y_test_flat = np.hstack(y_test)

# Cap the RUL at m = 125.

m = 125
y_train_flat = y_train_flat.clip(max=m)
y_test_flat = y_test_flat.clip(max=m)

# Cap sensor values at the 99th percentile to remove extreme spikes
lower = np.percentile(X_train_flat, 1, axis=0)
upper = np.percentile(X_train_flat, 99, axis=0)

X_train_flat = np.clip(X_train_flat, lower, upper)
X_test_flat = np.clip(X_test_flat, lower, upper)

scaler = MinMaxScaler()

final_data = X_train_flat.copy()
final_data[:, :] = scaler.fit_transform(final_data[:, :])

X_test_scaled = scaler.transform(X_test_flat)

model = RandomForestRegressor(n_estimators=100, max_depth=15, min_samples_leaf= 5, max_features='sqrt', random_state=42)

# Run 5-Fold Cross Validation
scores = cross_val_score(model, final_data, y_train_flat, cv=5, scoring='neg_root_mean_squared_error')

rmse_scores = -scores
print(f"Average RMSE: {rmse_scores.mean():.2f}")
print(f"Standard Deviation: {rmse_scores.std():.2f}")

model.fit(final_data, y_train_flat)

y_pred = model.predict(X_test_scaled)

# Calculate and print RMSE value
rmse = np.sqrt(mean_squared_error(y_test_flat, y_pred))
print(f'Model RMSE: {rmse}')

# Calculate and print R^2 Score
score = model.score(X_test_scaled, y_test_flat)
print(f'Model R^2 Score: {score}')

# Sanity Check:
self_score = model.score(final_data, y_train_flat)
print(f'Self Score (on training data): {self_score}')

plt.figure(figsize=(10, 6))
plt.scatter(y_test_flat, y_pred, alpha=0.5, color='blue', label='Predictions')

min_val = min(y_test_flat)
max_val = max(y_test_flat)
plt.plot([min_val, max_val], [min_val, max_val], color='red', linewidth=2, label='Perfect Fit')

plt.xlabel('Actual RUL (Cycles)')
plt.ylabel('Predicted RUL (Cycles)')
plt.title('Random Forest: Actual vs Predicted RUL')
plt.legend()
plt.grid(True)
plt.savefig('my_result.png')
print("Plot saved as my_result.png")

# Ploting for a perticular engine for the last 50 cycles
unit_id = 42
engine_data = all_engine_data[unit_id]       

final_data = engine_data.copy()
engine_data_42 = pd.DataFrame(final_data)
final_data = engine_data_42.tail(50).values

final_data[:, :] = scaler.transform(final_data[:, :])

predictions = model.predict(final_data)
predictions[predictions < 0] = 0 # Clip negatives

ruls_42 = ruls[unit_id][-50:]

plt.figure(figsize=(10, 6))
plt.plot(ruls_42, label='Actual RUL', color='orange', linewidth=3)
plt.plot(predictions, label='Predicted RUL', color='blue', linestyle='--', linewidth=2)
plt.title(f'Lifecycle of Engine {unit_id} (NumPy Version)')
plt.xlabel('Time (Cycles)')
plt.ylabel('RUL')
plt.legend()
plt.grid(True)
plt.savefig(f'numpy_engine_{unit_id}.png')
print(f"Plot saved for Engine {unit_id}")
