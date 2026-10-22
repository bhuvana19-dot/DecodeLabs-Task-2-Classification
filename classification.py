
from sklearn.datasets import load_iris

# Load the Iris flower dataset
iris = load_iris()

# Display the dataset information
print("Iris Flower Classification")
print("--------------------------")
print("Feature names:", iris.feature_names)
print("Flower names:", iris.target_names)
print("Total flower records:", len(iris.data))

from sklearn.model_selection import train_test_split

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    iris.data,
    iris.target,
    test_size=0.2,
    random_state=42
)

print("\nDataset splitting results")
print("Training records:", len(X_train))
print("Testing records:", len(X_test))

from sklearn.tree import DecisionTreeClassifier

# Create the classification model
model = DecisionTreeClassifier(random_state=42)

# Train the model using training data
model.fit(X_train, y_train)

print("\nAI model training completed!")

from sklearn.metrics import accuracy_score

# Predict flower categories for the test data
y_pred = model.predict(X_test)

# Calculate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Testing Results")
print("Correct predictions:", sum(y_pred == y_test), "out of", len(y_test))
print("Model accuracy:", round(accuracy * 100, 2), "%")

# Predict the species of a new flower
new_flower = [[5.1, 3.5, 1.4, 0.2]]

prediction = model.predict(new_flower)

flower_name = iris.target_names[prediction[0]]

print("\nNew Flower Prediction")
print("Predicted flower:", flower_name)
