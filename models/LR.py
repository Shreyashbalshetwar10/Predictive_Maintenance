import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

data_1 = np.loadtxt(r'/home/shreyash/Projects/Predictive_Maintenance/CMaps/train_FD002.txt')

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


model = LinearRegression()

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

df_temp = pd.DataFrame(X_train_flat)
df_temp['RUL'] = y_train_flat

correlations = df_temp.corrwith(df_temp['RUL'])

print(correlations)

# Based on correlation, we choose sensors to keep
sensors_to_remove = []

for i in range(no_clmns := X_train_flat.shape[1]):
    if abs(correlations[i]) < 0.1 or np.isnan(correlations[i]) and i not in sensors_to_remove:
        sensors_to_remove.append(i)
        #Remove the identified sensors from both training and testing data
        print(f"Removing sensor {i} with correlation {correlations[i]:.4f}")

x_train_reduced = np.delete(X_train_flat, sensors_to_remove, axis=1)
x_test_reduced = np.delete(X_test_flat, sensors_to_remove, axis=1)

features = x_train_reduced

scaler = MinMaxScaler()

final_data = x_train_reduced.copy()
final_data[:, :] = scaler.fit_transform(final_data[:, :])

X_test_scaled = scaler.transform(x_test_reduced)

model.fit(final_data, y_train_flat)

y_pred = model.predict(X_test_scaled)


# Plotting actual vs predicted RUL for the test set
plt.figure(figsize=(10, 6))
plt.scatter(y_test_flat, y_pred, alpha=0.5, color='blue', label='Predictions')

# Draw a red diagonal line (Perfect Prediction)
# We use the min and max of the actual values to define the line
min_val = min(y_test_flat)
max_val = max(y_test_flat)
plt.plot([min_val, max_val], [min_val, max_val], color='red', linewidth=2, label='Perfect Fit')

plt.xlabel('Actual RUL (Cycles)')
plt.ylabel('Predicted RUL (Cycles)')
plt.title('Linear Regression: Actual vs Predicted RUL')
plt.legend()
plt.grid(True)
plt.savefig('my_result_2.png')
print("Plot saved as my_result_2.png")



unit_id = 42
engine_data = all_engine_data[unit_id]

engine_data = np.delete(engine_data, sensors_to_remove, axis=1)

final_data = engine_data.copy()
engine_data_42 = pd.DataFrame(final_data)
final_data = engine_data_42.tail(50).values

final_data[:, :] = scaler.transform(final_data[:, :])

predictions = model.predict(final_data)
predictions[predictions < 0] = 0 # Clip negatives

ruls_42 = ruls[unit_id][-50:]  # Last 50 RUL values for engine 42

plt.figure(figsize=(10, 6))
plt.plot(ruls_42, label='Actual RUL', color='orange', linewidth=3)
plt.plot(predictions, label='Predicted RUL', color='blue', linestyle='--', linewidth=2)
plt.title(f'Lifecycle of Engine {unit_id} (NumPy Version)')
plt.xlabel('Time (Cycles)')
plt.ylabel('RUL')
plt.legend()
plt.grid(True)
plt.savefig(f'numpy_engine_2_{unit_id}.png')
print(f"Plot saved for Engine {unit_id}")

results_df = pd.DataFrame({'Actual': y_test_flat, 'Predicted': y_pred})

subset = results_df.tail(50).copy() # Use .copy() to avoid warnings

subset = subset.sort_values(by='Actual').reset_index(drop=True)

subset['Predicted'] = subset['Predicted'].clip(lower=0) # Clip negatives

plt.figure(figsize=(12, 6))

# Plot Actual (Orange Line)
plt.plot(subset.index, subset['Actual'], label='Actual RUL', color='orange', linewidth=2)

# Plot Predicted (Blue Dots/Lines)
# We use a Scatter plot for predictions to see how they cluster around the line
plt.scatter(subset.index, subset['Predicted'], label='Predicted RUL', color='blue', alpha=0.7, s=20)
# Optional: Add error bars or vertical lines connecting Prediction to Actual
plt.vlines(subset.index, subset['Actual'], subset['Predicted'], colors='gray', linestyles='dotted', alpha=0.5)

plt.xlabel('Sample Index (Sorted by RUL)')
plt.ylabel('RUL (Cycles)')
plt.title('Linear Regression: Predictions vs Actual (Sorted)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.savefig('my_result_sorted_2.png')
print("Plot saved as my_result_sorted_2.png")

score = model.score(X_test_scaled, y_test_flat)
print(f'Model R^2 Score: {score}')

rmse = np.sqrt(mean_squared_error(y_test_flat, y_pred))
print(f'Model RMSE: {rmse}')
