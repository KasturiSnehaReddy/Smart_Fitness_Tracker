import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt

# Load and preprocess data
data = pd.read_csv("calories.csv")

# Encode categorical column
le = LabelEncoder()
data["Gender"] = le.fit_transform(data["Gender"])

X = data.drop(["User_ID", "Calories"], axis=1).values
y = data["Calories"].values.reshape(-1, 1)

scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split into train, val, test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Initialize weights
input_size = X_train.shape[1]
hidden_size = 10
output_size = 1

W1 = np.random.randn(input_size, hidden_size) * 0.01
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size) * 0.01
b2 = np.zeros((1, output_size))

# Activation
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

# Loss
def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Forward
def forward_pass(X):
    Z1 = np.dot(X, W1) + b1
    A1 = relu(Z1)
    Z2 = np.dot(A1, W2) + b2
    return Z1, A1, Z2

# Backward
def backward_pass(X, y, Z1, A1, Z2):
    m = y.shape[0]
    dZ2 = (Z2 - y) / m
    dW2 = np.dot(A1.T, dZ2)
    db2 = np.sum(dZ2, axis=0, keepdims=True)
    
    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * relu_derivative(Z1)
    dW1 = np.dot(X.T, dZ1)
    db1 = np.sum(dZ1, axis=0, keepdims=True)
    
    return dW1, db1, dW2, db2


# Training with early stopping + checkpoint saving
learning_rate = 0.001
epochs = 3000
best_val_loss = float("inf")
patience = 50
counter = 0
train_losses = []
val_losses = []

for epoch in range(epochs):
    Z1, A1, Z2 = forward_pass(X_train)
    loss = mse_loss(y_train, Z2)
    train_losses.append(loss)

    dW1, db1, dW2, db2 = backward_pass(X_train, y_train, Z1, A1, Z2)
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    # Evaluation on validation set
    _, _, val_pred = forward_pass(X_val)
    val_loss = mse_loss(y_val, val_pred)
    val_losses.append(val_loss)

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Train Loss: {loss:.4f}, Val Loss: {val_loss:.4f}")

    # Early stopping check + save best checkpoint
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        np.savez("best_model.npz", W1=W1, b1=b1, W2=W2, b2=b2)  # ✅ Save checkpoint
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print(f"\nStopped early at epoch {epoch} due to no improvement in validation loss.")
            break

# Plot losses
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Load best weights
checkpoint = np.load("best_model.npz")
W1, b1, W2, b2 = checkpoint["W1"], checkpoint["b1"], checkpoint["W2"], checkpoint["b2"]

# Final test evaluation
_, _, test_pred = forward_pass(X_test)
test_loss = mse_loss(y_test, test_pred)
print(f"\nFinal Test Loss: {test_loss:.4f}")
