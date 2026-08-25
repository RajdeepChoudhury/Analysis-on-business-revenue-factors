
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# File paths
INPUT_FILE = "Business Revenue Factors Criticality.xlsx"
OUTPUT_FILE = "business_revenue_factor_clusters.xlsx"
PLOT_FILE = "factor_clusters.png"

# Load data
df = pd.read_excel(INPUT_FILE)

# Clean column names
df.columns = [c.strip().replace(" ", "_").replace(".", "").replace("/", "_") for c in df.columns]

# Convert numeric columns safely
for col in ["Mean", "Std_Dev", "Mean_Rank", "Skewness", "Kurtosis"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col].replace("–", np.nan), errors="coerce")

# Features for clustering
features = ["Mean", "Std_Dev", "Skewness", "Kurtosis"]
X = df[features]

# Pipeline: impute -> scale -> cluster
k = 3
pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("kmeans", KMeans(n_clusters=k, random_state=42, n_init=10))
])

cluster_raw = pipe.fit_predict(X)
df["Cluster_raw"] = cluster_raw

# Reorder cluster labels by average criticality (Mean): lower Mean = more critical
centers = pipe.named_steps["kmeans"].cluster_centers_
order = np.argsort(centers[:, 0])  # ascending standardized Mean
mapping = {old: new + 1 for new, old in enumerate(order)}
df["Cluster"] = df["Cluster_raw"].map(mapping)

# Cluster summary
summary = (
    df.groupby("Cluster")[["Mean", "Std_Dev", "Mean_Rank"]]
    .agg(["mean", "count"])
)

# Save results
with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
    df.to_excel(writer, index=False, sheet_name="Clustered_Factors")
    summary.to_excel(writer, sheet_name="Cluster_Summary")

# Print evaluation
X_imp = SimpleImputer(strategy="median").fit_transform(X)
X_scaled = StandardScaler().fit_transform(X_imp)
sil = silhouette_score(X_scaled, df["Cluster_raw"])

print("Silhouette score:", round(float(sil), 4))
print("\nCluster assignments:")
print(df[["Factors", "Mean", "Std_Dev", "Mean_Rank", "Cluster"]].sort_values(["Cluster", "Mean"]))

print("\nCluster summary:")
print(summary)

# Visualize
plt.figure(figsize=(12, 6))
colors = {1: "tab:red", 2: "tab:orange", 3: "tab:green"}
for cl in sorted(df["Cluster"].unique()):
    sub = df[df["Cluster"] == cl]
    plt.scatter(sub["Factors"], sub["Mean"], s=120, label=f"Cluster {cl}", color=colors.get(cl))
plt.axhline(0, linestyle="--", linewidth=1)
plt.title("Business Revenue Factors Clustered by Criticality")
plt.xlabel("Factors")
plt.ylabel("Mean criticality")
plt.xticks(rotation=90)
plt.legend()
plt.tight_layout()
plt.savefig(PLOT_FILE, dpi=200)
plt.show()
