import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, f1_score
import joblib

# Load dataset
data = pd.read_csv("train.csv")

# Feature Engineering: Family Size and IsAlone
data["FamilySize"] = data["SibSp"] + data["Parch"]
data["IsAlone"] = (data["FamilySize"] == 0).astype(int)

# Select features and drop missing values
features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "FamilySize", "IsAlone"]
data = data[features + ["Survived"]].dropna()

# One-hot encode categorical variables
data = pd.get_dummies(data, columns=["Sex", "Embarked"])

# Split into X and y
X = data.drop("Survived", axis=1)
y = data["Survived"]

# Save feature columns to ensure consistent order during inference
joblib.dump(X.columns.tolist(), "model_features.pkl")

# Train-test split with fixed random seed
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Define models with fixed random_state for determinism
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42)
}

# Train, evaluate, and save each model
accuracies = {}
precisions = {}
f1_scores = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    accuracies[name] = acc
    precisions[name] = prec
    f1_scores[name] = f1

    joblib.dump(model, f"{name}_titanic_model.pkl")

    print(f"{name} accuracy: {acc:.4f}")
    print("Precision:", prec)
    print("F1 Score:", f1)

# Save accuracy report
accuracy_df = pd.DataFrame.from_dict(accuracies, orient='index', columns=['Accuracy'])
accuracy_df.to_csv("model_accuracies.csv")

# Plot accuracy comparison
plt.figure(figsize=(8, 5))
sns.barplot(x=accuracy_df.index, y=accuracy_df["Accuracy"], palette="viridis")
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.xlabel("Model")
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Plot all metrics comparison
metrics_df = pd.DataFrame({
    "Accuracy": accuracies,
    "Precision": precisions,
    "F1 Score": f1_scores
}).T

metrics_df.plot(kind="bar", figsize=(10, 6), colormap="Set2")
plt.title("Model Performance Comparison")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# Feature importance plot for Random Forest
importances = models["Random Forest"].feature_importances_
feature_names = X.columns

plt.figure(figsize=(8, 5))
sns.barplot(x=importances, y=feature_names)
plt.title("Feature Importance from Random Forest Classifier")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("feature_importance_rf.png")
plt.close()
