import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt

# Load and preprocess data
data = pd.read_csv(r"src/calories.csv")

# Encode categorical column
le = LabelEncoder()
data["Gender"] = le.fit_transform(data["Gender"])

X = data.drop(["User_ID", "Calories"], axis=1).values
y = data["Calories"].values.reshape(-1, 1)

scaler = StandardScaler()
X = scaler.fit_transform(X)

# Save the scaler parameters for use in the app
np.savez("checkpoints/scaler.npz", mean=scaler.mean_, scale=scaler.scale_)

# Split into train, val, test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Initialize weights with better initialization (Xavier/Glorot)
input_size = X_train.shape[1]
hidden_size = 16  # Increased hidden layer size
output_size = 1

def initialize_weights(input_size, hidden_size, output_size):
    # Xavier/Glorot initialization for better training
    W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
    b2 = np.zeros((1, output_size))
    return W1, b1, W2, b2

# Activation
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

# Loss
def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Forward
def forward_pass(X, W1, b1, W2, b2):
    Z1 = np.dot(X, W1) + b1
    A1 = relu(Z1)
    Z2 = np.dot(A1, W2) + b2
    return Z1, A1, Z2

# Backward
def backward_pass(X, y, Z1, A1, Z2, W1, W2):
    m = y.shape[0]
    dZ2 = (Z2 - y) / m
    dW2 = np.dot(A1.T, dZ2)
    db2 = np.sum(dZ2, axis=0, keepdims=True)
    
    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * relu_derivative(Z1)
    dW1 = np.dot(X.T, dZ1)
    db1 = np.sum(dZ1, axis=0, keepdims=True)
    
    return dW1, db1, dW2, db2

# Training with multiple runs to find best model
best_val_loss = float("inf")
best_weights = None
best_run = 0

print("Training multiple models to find the best one...")

for run in range(5):  # Train 5 different models
    print(f"\n=== Training Run {run + 1}/5 ===")
    
    # Initialize weights for this run
    W1, b1, W2, b2 = initialize_weights(input_size, hidden_size, output_size)
    
    # Training parameters
    learning_rate = 0.001
    epochs = 5000  # More epochs
    patience = 100  # More patience
    counter = 0
    train_losses = []
    val_losses = []
    
    run_best_val_loss = float("inf")
    run_best_weights = None
    
    for epoch in range(epochs):
        # Forward pass
        Z1, A1, Z2 = forward_pass(X_train, W1, b1, W2, b2)
        loss = mse_loss(y_train, Z2)
        train_losses.append(loss)

        # Backward pass
        dW1, db1, dW2, db2 = backward_pass(X_train, y_train, Z1, A1, Z2, W1, W2)
        
        # Update weights
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2

        # Evaluation on validation set
        _, _, val_pred = forward_pass(X_val, W1, b1, W2, b2)
        val_loss = mse_loss(y_val, val_pred)
        val_losses.append(val_loss)

        if epoch % 500 == 0:
            print(f"Epoch {epoch}, Train Loss: {loss:.4f}, Val Loss: {val_loss:.4f}")

        # Early stopping check for this run
        if val_loss < run_best_val_loss:
            run_best_val_loss = val_loss
            run_best_weights = (W1.copy(), b1.copy(), W2.copy(), b2.copy())
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Run {run + 1} stopped early at epoch {epoch} with val loss: {run_best_val_loss:.4f}")
                break
    
    # Check if this run is better than the overall best
    if run_best_val_loss < best_val_loss:
        best_val_loss = run_best_val_loss
        best_weights = run_best_weights
        best_run = run + 1
        print(f"🎉 New best model found! Run {run + 1} with val loss: {best_val_loss:.4f}")

print(f"\n🏆 Best model from run {best_run} with validation loss: {best_val_loss:.4f}")

# Load best weights
W1, b1, W2, b2 = best_weights

# Save the best model
np.savez("checkpoints/best_model.npz", W1=W1, b1=b1, W2=W2, b2=b2)

# Final test evaluation
_, _, test_pred = forward_pass(X_test, W1, b1, W2, b2)
test_loss = mse_loss(y_test, test_pred)
print(f"\n📊 Final Test Loss: {test_loss:.4f}")

# Calculate R-squared score
from sklearn.metrics import r2_score
r2 = r2_score(y_test, test_pred)
print(f"📈 R-squared Score: {r2:.4f}")

# Plot training history for the best run
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss', alpha=0.7)
plt.plot(val_losses, label='Validation Loss', alpha=0.7)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title(f'Training History (Best Run {best_run})')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.scatter(y_test, test_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Calories')
plt.ylabel('Predicted Calories')
plt.title(f'Prediction vs Actual (R² = {r2:.4f})')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('src/training_results.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\n✅ Best model saved with test loss: {test_loss:.4f}")
print(f"📁 Model saved to: checkpoints/best_model.npz")
print(f"📁 Scaler saved to: checkpoints/scaler.npz")
print(f"📊 Training plot saved to: src/training_results.png")
