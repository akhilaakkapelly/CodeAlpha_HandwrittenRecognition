# CodeAlpha_HandwrittenRecognition

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load handwritten digit dataset
digits = load_digits()

X = digits.data
y = digits.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train ML model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# Predict digits
predictions = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)

print("Handwritten Digit Recognition")
print("Model Accuracy:", accuracy * 100, "%")

# Display sample predictions
for i in range(5):
    print("Actual:", y_test[i],
          "Predicted:", predictions[i])

print("\nClassification Report:")
print(classification_report(y_test, predictions))
