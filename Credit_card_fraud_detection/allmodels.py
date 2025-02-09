import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import matthews_corrcoef, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from joblib import Parallel, delayed

# Load the dataset
df = pd.read_csv("creditcard.csv")
X = df.drop(["Class"], axis=1)
y = df["Class"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define models
models = {
    "Logistic Regression": LogisticRegression(),
    "k-Nearest Neighbors": KNeighborsClassifier(),
    "Support Vector Machine": SVC(),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Extra Trees": ExtraTreesClassifier(random_state=42),
}

# Define rectifiers
rectifiers = {
    "No Rectifier": None,
    "SMOTE": SMOTE(),
    "Near Miss": NearMiss(),
}

# Evaluate model function
def evaluate_model(model_name, model, rectifier_name, rectifier, X_train, y_train, X_test, y_test):
    if rectifier:
        X_train_bal, y_train_bal = rectifier.fit_resample(X_train, y_train)
    else:
        X_train_bal, y_train_bal = X_train, y_train

    # Train model
    model.fit(X_train_bal, y_train_bal)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    mcc = matthews_corrcoef(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, output_dict=False)

    return {
        "Model": model_name,
        "Rectifier": rectifier_name,
        "MCC": mcc,
        "Confusion Matrix": cm,
        "Classification Report": cr,
    }

# Parallel execution for all combinations
results = Parallel(n_jobs=-1)(
    delayed(evaluate_model)(
        model_name, model, rectifier_name, rectifier, X_train, y_train, X_test, y_test
    )
    for model_name, model in models.items()
    for rectifier_name, rectifier in rectifiers.items()
)

# Convert results to a DataFrame for easier visualization
results_df = pd.DataFrame(results)

# Save results locally for analysis
results_df.to_csv("model_comparison_results.csv", index=False)
print("Results saved to model_comparison_results.csv")
