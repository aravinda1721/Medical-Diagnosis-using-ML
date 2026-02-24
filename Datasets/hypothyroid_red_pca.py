import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv(r"C:\Users\candy\Desktop\ML_project\Datasets\preprocessed_hypothyroid.csv")

# Separate features and target
X = df.drop(columns=["binaryClass"])
y = df["binaryClass"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save scaler
joblib.dump(scaler, r"C:\Users\candy\Desktop\ML_project\Models\hypothyroid_scaler.pkl")

# PCA full analysis (variance plot)
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
plt.title("PCA – Hypothyroid Dataset")
plt.grid(True)
plt.savefig(r"C:\Users\candy\Desktop\ML_project\Datasets\hypothyroid_pca_variance_plot.png")
plt.close()

# PCA for 95% variance reduction
pca = PCA(n_components=0.95)
X_reduced = pca.fit_transform(X_scaled)

# Save PCA model
joblib.dump(pca, r"C:\Users\candy\Desktop\ML_project\Models\hypothyroid_pca.pkl")

# Create reduced dataset
reduced_df = pd.DataFrame(
    X_reduced,
    columns=[f"PC{i+1}" for i in range(X_reduced.shape[1])]
)
reduced_df["binaryClass"] = y

output_path = r"C:\Users\candy\Desktop\ML_project\Datasets\hypothyroid_reduced_pca.csv"
reduced_df.to_csv(output_path, index=False)

print("\n🎉 PCA Completed Successfully!")
print("Original Feature Count:", X.shape[1])
print("Reduced Feature Count:", X_reduced.shape[1])
print("Saved:", output_path)
print("Scaler + PCA Saved Successfully")
