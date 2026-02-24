import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
import joblib

df = pd.read_csv(r"C:\Users\candy\Desktop\ML_project\Datasets\lung_reduced_pca.csv")

X = df.drop(columns=["lung_cancer"])
y = df["lung_cancer"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=3000),
    "SVM": SVC(probability=True, kernel="rbf"),
    "Random Forest": RandomForestClassifier(n_estimators=200),
    "Gradient Boosting": GradientBoostingClassifier(),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "XGBoost": XGBClassifier(eval_metric='logloss')
}

metrics_table = []
best_model = None
best_model_name = None
best_f1 = 0

print("\n=================== TRAINING MODELS (LUNG CANCER) ===================\n")

for name, model in models.items():
    print(f"\n------ Training: {name} ------")
    
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    # Compute metrics
    acc = accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds, zero_division=0)
    recall = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    
    metrics_table.append([name, acc, precision, recall, f1])

    print("\nClassification Report:")
    print(classification_report(y_test, preds, zero_division=0))

    # Pick best model using F1-score
    if f1 > best_f1:
        best_f1 = f1
        best_model_name = name
        best_model = model

results_df = pd.DataFrame(
    metrics_table, 
    columns=["Model", "Accuracy", "Precision", "Recall", "F1-Score"]
)

print("\n=================== RESULTS SUMMARY (LUNG CANCER) ===================\n")
print(results_df.to_string(index=False))

save_path = r"C:\Users\candy\Desktop\ML_project\Models\lungs_best_model.pkl"
joblib.dump(best_model, save_path)

print("\n======================================================")
print(" BEST MODEL FOR LUNG CANCER:", best_model_name)
print(f" F1-Score: {best_f1:.4f}")
print(" Model saved at:", save_path)
print("======================================================\n")
