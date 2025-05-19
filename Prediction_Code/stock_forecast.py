import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from datetime import datetime

# Directories
data_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../Preprocessed_Data"))

plot_dir = "plots_50" # change the name of plots directory according to the number of epochs to differentiate
model_dir = "saved_models"
os.makedirs(plot_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

force_retrain = False  # Change to True when you want a new model

# Preprocessing function
def preprocess_stock_data(file_path):
    df = pd.read_csv(file_path)
    df["Date"] = pd.to_datetime(df["Date"], format="%d-%b-%Y")
    df.set_index("Date", inplace=True)
    df = df[["Open Price", "High Price", "Low Price", "Close Price"]]
    df = df.sort_index()
    return df

# Create sequences
def create_sequences(data, time_step=50):
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:i+time_step])
        y.append(data[i+time_step, 3])
    return np.array(X), np.array(y)

# Save model
def save_model(model):
    filename = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5"
    model.save(os.path.join(model_dir, filename))
    print(f"Model saved as {filename}")

# Load most recent model or the required model
def load_latest_model():
    models = sorted(glob.glob(os.path.join(model_dir, "*.h5")), reverse=True)
    if models:
        print(f"Loaded model: {os.path.basename(models[2])}") #Choose models based on index
        return load_model(models[2]) #Use the same model index as the previous line
    return None

# Load and prepare all data
all_X, all_y = [], []
time_step = 50
file_paths = glob.glob(os.path.join(data_root, "*/*.csv"))
scalers = {}

print("Processing files...")
for path in file_paths:
    df = preprocess_stock_data(path)
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df)
    scalers[path] = scaler
    X, y = create_sequences(df_scaled, time_step)
    all_X.append(X)
    all_y.append(y)

# Concatenate all sequences
X_all = np.concatenate(all_X, axis=0)
y_all = np.concatenate(all_y, axis=0)

# Train-test split
split_index = int(len(X_all) * 0.8)
X_train, X_test = X_all[:split_index], X_all[split_index:]
y_train, y_test = y_all[:split_index], y_all[split_index:]

# Either use existing Model or create a new Model:
if not force_retrain:
    model = load_latest_model()
else:
    model = None
# # Load or train model

if model is None:
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(time_step, 4)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    learning_rate = 0.001
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))
    save_model(model)

# Evaluation metrics
metrics = []

print("\nEvaluating on individual stock files...")

for path in file_paths:
    file_name = os.path.basename(path).replace(".csv", "")
    df = preprocess_stock_data(path)
    scaler = scalers[path]
    df_scaled = scaler.transform(df)
    X, y = create_sequences(df_scaled, time_step)

    split_index = int(len(X) * 0.8)
    X_eval, y_eval = X[split_index:], y[split_index:]

    y_pred = model.predict(X_eval)
    
    y_pred_rescaled = scaler.inverse_transform(np.concatenate((np.zeros((len(y_pred), 3)), y_pred), axis=1))[:, 3]
    y_eval_rescaled = scaler.inverse_transform(np.concatenate((np.zeros((len(y_eval), 3)), y_eval.reshape(-1, 1)), axis=1))[:, 3]

    # Metrics
    tolerance = 0.05
    accuracy_mask = np.abs(y_eval_rescaled - y_pred_rescaled) <= (tolerance * y_eval_rescaled)
    custom_acc = np.mean(accuracy_mask) * 100

    metrics.append([file_name, custom_acc])

    # Plot
    plt.figure(figsize=(12, 5))
    plt.plot(y_eval_rescaled, label="Actual Price")
    plt.plot(y_pred_rescaled, label="Predicted Price")
    plt.title(f"{file_name} - Prediction")
    plt.legend()
    plt.savefig(os.path.join(plot_dir, f"{file_name}_prediction.png"))
    plt.close()

# Convert to DataFrame and save
metrics_df = pd.DataFrame(metrics, columns=["Stock", "Custom Accuracy (%)"])
print(metrics_df)

# Save metrics to csv
metrics_df.to_csv("evaluation_metrics.csv", index=False)

# Print average metrics
avg_metrics = metrics_df[["Custom Accuracy (%)"]].mean()
print(avg_metrics)
