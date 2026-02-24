# --------------------------------------------
# PCA Feature Reduction for Diabetes + Save Scaler + PCA
# --------------------------------------------

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib

# Load data
data = pd.read_csv(r"C:\Users\candy\Desktop\ML_project\Datasets\diabetes_data.csv")

# Separate features and target
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA (95% variance)
pca_95 = PCA(n_components=0.95)
X_reduced = pca_95.fit_transform(X_scaled)

# Save scaler and PCA model
joblib.dump(scaler, r"C:\Users\candy\Desktop\ML_project\Models\diabetes_scaler.pkl")
joblib.dump(pca_95, r"C:\Users\candy\Desktop\ML_project\Models\diabetes_pca.pkl")

# Save PCA dataset
import pandas as pd
reduced_df = pd.DataFrame(
    X_reduced,
    columns=[f"PC{i+1}" for i in range(X_reduced.shape[1])]
)
reduced_df["Target"] = y

reduced_df.to_csv(r"C:\Users\candy\Desktop\ML_project\Datasets\diabetes_reduced_pca.csv", index=False)

print("Scaler and PCA saved successfully!")
print("Reduced PCA dataset generated!")
