import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv(r"C:\Users\candy\Desktop\ML_project\Datasets\parkinson_data.csv")

# Remove name column
df = df.drop(columns=["name"])

# Separate features and target
X = df.drop(columns=["status"])
y = df["status"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save the scaler
joblib.dump(scaler, r"C:\Users\candy\Desktop\ML_project\Models\parkinson_scaler.pkl")

# Show cumulative variance (optional)
pca_all = PCA()
pca_all.fit(X_scaled)

plt.figure(figsize=(8,5))
plt.plot(
    range(1, len(pca_all.explained_variance_ratio_) + 1),
    pca_all.explained_variance_ratio_.cumsum(),
    marker='o'
)
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA – Parkinson's Dataset")
plt.grid(True)
plt.savefig(r"C:\Users\candy\Desktop\ML_project\Datasets\parkinson_pca_variance_plot.png")
plt.close()

# PCA with 95% variance
pca = PCA(n_components=0.95)
X_reduced = pca.fit_transform(X_scaled)

# Save PCA model
joblib.dump(pca, r"C:\Users\candy\Desktop\ML_project\Models\parkinson_pca.pkl")

# Build reduced dataframe
reduced_df = pd.DataFrame(
    X_reduced,
    columns=[f"PC{i+1}" for i in range(X_reduced.shape[1])]
)

# Add target back
reduced_df["status"] = y

# Save PCA-reduced dataset
output_path = r"C:\Users\candy\Desktop\ML_project\Datasets\parkinson_reduced_pca.csv"
reduced_df.to_csv(output_path, index=False)

print("PCA Completed Successfully!")
print("Original Features:", X.shape[1])
print("Reduced Features:", X_reduced.shape[1])
print("Saved PCA Dataset:", output_path)
print("Scaler + PCA Saved!")
