import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib

# Load your dataset
df = pd.read_csv(r"C:\Users\candy\Desktop\ML_project\Datasets\heart_disease_data.csv")

# Separate features and target
X = df.drop(columns=["target"])
y = df["target"]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA with 95% variance
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)

# Save scaler and PCA
joblib.dump(scaler, r"C:\Users\candy\Desktop\ML_project\Models\heart_scaler.pkl")
joblib.dump(pca, r"C:\Users\candy\Desktop\ML_project\Models\heart_pca.pkl")

# Save PCA reduced dataset
reduced_df = pd.DataFrame(
    X_pca,
    columns=[f"PC{i+1}" for i in range(X_pca.shape[1])]
)
reduced_df["target"] = y

reduced_df.to_csv(
    r"C:\Users\candy\Desktop\ML_project\Datasets\heart_reduced_pca.csv",
    index=False
)

print("Heart scaler + PCA saved successfully!")
print("Reduced PCA file updated!")
