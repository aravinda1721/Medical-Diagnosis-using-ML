# ---------------------------------------------------------
# PCA Feature Reduction for Lung Cancer Dataset
# ---------------------------------------------------------

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import joblib

# ---------------------------------------------------------
# 1. Load the cleaned dataset
# ---------------------------------------------------------
df = pd.read_csv(r"C:\Users\candy\Desktop\ML_project\Datasets\preprocessed_lungs_data.csv")

# Separate features and target
X = df.drop(columns=["LUNG_CANCER"])
y = df["LUNG_CANCER"]

# ---------------------------------------------------------
# 2. Standardize the features
# ---------------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ---------------------------------------------------------
# 3. Apply PCA (Full for variance graph)
# ---------------------------------------------------------
pca = PCA()
pca.fit(X_scaled)

# ---------------------------------------------------------
# 4. Plot explained variance curve
# ---------------------------------------------------------
plt.figure(figsize=(8, 5))
plt.plot(
    range(1, len(pca.explained_variance_ratio_) + 1),
    pca.explained_variance_ratio_.cumsum(),
    marker='o'
)
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA - Lung Cancer Dataset Variance Curve")
plt.grid(True)

plot_path = r"C:\Users\candy\Desktop\ML_project\Datasets\lung_pca_variance_plot.png"
plt.savefig(plot_path)
plt.close()

# ---------------------------------------------------------
# 5. Reduce dimensions to keep 95% variance
# ---------------------------------------------------------
pca_95 = PCA(n_components=0.95)
X_reduced = pca_95.fit_transform(X_scaled)

# ---------------------------------------------------------
# 6. Save reduced dataset
# ---------------------------------------------------------
reduced_df = pd.DataFrame(
    X_reduced,
    columns=[f"PC{i+1}" for i in range(X_reduced.shape[1])]
)

# Append Target
reduced_df["lung_cancer"] = y

output_path = r"C:\Users\candy\Desktop\ML_project\Datasets\lung_reduced_pca.csv"
reduced_df.to_csv(output_path, index=False)

joblib.dump(scaler, r"C:\Users\candy\Desktop\ML_project\Models\lung_scaler.pkl")
joblib.dump(pca_95, r"C:\Users\candy\Desktop\ML_project\Models\lung_pca.pkl")


# ---------------------------------------------------------
# 7. Display summary
# ---------------------------------------------------------
print("🎉 PCA Completed Successfully!")
print("Original features:", X.shape[1])
print("Reduced features (95% variance):", X_reduced.shape[1])
print("📁 Reduced dataset saved as:", output_path)
print("📊 PCA variance plot saved at:", plot_path)

print("Scaler & PCA Saved Successfully!")
