# Imports
from tensorflow import keras 
import pandas as pd 
import numpy as np 
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt 
import seaborn as sns 
import os 
from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



data = pd.read_csv("MicrosoftStock.csv")
print(data.head())
print(data.info())
print(data.describe())

data['date'] = pd.to_datetime(data['date'])
data['direction'] = (data['close'].shift(-1) > data['close']).astype(int)


# Initial Data Visualization
# Plot 1 - Open and Close Prices of time
plt.figure(figsize=(12,6))
plt.plot(data['date'], data['open'], label="Open",color="blue")
plt.plot(data['date'], data['close'], label="Close",color="red")
plt.title("Open-Close Price over Time")
plt.legend()
# plt.show()

# Plot 2 - Trading Volume (check for outliers)
plt.figure(figsize=(12,6))
plt.plot(data['date'],data['volume'],label="Volume",color="orange")
plt.title("Stock Volume over Time")
# plt.show()


# Drop non-numeric columns
numeric_data = data.select_dtypes(include=["int64","float64"])

# Plot 3 - Check for correlation between features
plt.figure(figsize=(8,6))
sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
# plt.show()


# Convert the Data into Date time then create a date filter
prediction = data.loc[
    (data['date'] > datetime(2013,1,1)) &
    (data['date'] < datetime(2018,1,1))
]

plt.figure(figsize=(12,6))
plt.plot(data['date'], data['close'],color="blue")
plt.xlabel("Date")
plt.ylabel("Close")
plt.title("Price over time")


# Prepare for the LSTM Model (Sequential)
dataset = data[['close', 'direction']].dropna().values

training_data_len = int(np.ceil(len(dataset) * 0.95))
scaler = StandardScaler()
scaled_data = scaler.fit_transform(dataset[:training_data_len, 0].reshape(-1, 1))  # Only scale 'close'

X_train, y_train = [], []

for i in range(60, len(scaled_data)):
    X_train.append(scaled_data[i-60:i, 0])
    y_train.append(dataset[i, 1])  # direction label

X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


# Build the Model
model = keras.models.Sequential()

# First Layer
model.add(keras.layers.LSTM(64, return_sequences=True, input_shape=(X_train.shape[1],1)))

# Second Layer
model.add(keras.layers.LSTM(64, return_sequences=False))

# 3rd Layer (Dense)
model.add(keras.layers.Dense(128, activation="relu"))

# 4th Layer (Dropout)
model.add(keras.layers.Dropout(0.5))

# Final Output Layer
model.add(keras.layers.Dense(1, activation="sigmoid"))

model.summary()
model.compile(optimizer="adam",
              loss="binary_crossentropy",
              metrics=["accuracy"])


training = model.fit(X_train, y_train, epochs=20, batch_size=32)

# Prep the test data
test_data = scaler.transform(dataset[training_data_len - 60:, 0].reshape(-1, 1))
X_test, y_test = [], dataset[training_data_len:, 1]

for i in range(60, len(test_data)):
    X_test.append(test_data[i-60:i, 0])

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Make a Prediction
predictions = model.predict(X_test)
predicted_classes = (predictions > 0.5).astype(int)

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
print("Confusion Matrix:\n", confusion_matrix(y_test, predicted_classes))
print(f"Accuracy: {accuracy_score(y_test, predicted_classes)}")
print(f"Precision: {precision_score(y_test, predicted_classes)}")
print(f"Recall: {recall_score(y_test, predicted_classes)}")




plt.figure(figsize=(12,8))
plt.plot(data['date'][:training_data_len], dataset[:training_data_len,0], label="Train (Actual)", color='blue')
plt.plot(data['date'][training_data_len:], dataset[training_data_len:,0], label="Test (Actual)", color='orange')
plt.title("Stock Close Price")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.legend()
plt.show()