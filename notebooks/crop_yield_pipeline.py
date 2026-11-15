"""
Crop Yield Prediction Under Changing Climate Conditions
End-to-End Machine Learning & Analytics Pipeline

Dataset: FAO (yield, pesticide use) & World Bank (rainfall, temperature)
Merged records: 28,242 rows across 101 countries and 10 staple crops (1990-2013).
"""

import json
import os
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110

# Dynamic directory resolution
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "notebooks" else SCRIPT_DIR

DATA_PATH = PROJECT_ROOT / "data" / "yield_df.csv"
FIG_DIR = PROJECT_ROOT / "figures"
MODEL_DIR = PROJECT_ROOT / "models"
EXPORT_DIR = PROJECT_ROOT / "exports"

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(EXPORT_DIR, exist_ok=True)

print(f"Project root: {PROJECT_ROOT}")
print(f"Loading data from: {DATA_PATH}")

# %% 1. Load & Clean Data
df = pd.read_csv(DATA_PATH, index_col=0)
df = df.rename(columns={
    "Area": "country",
    "Item": "crop",
    "Year": "year",
    "hg/ha_yield": "yield_hg_ha",
    "average_rain_fall_mm_per_year": "rainfall_mm",
    "pesticides_tonnes": "pesticides_tonnes",
    "avg_temp": "avg_temp_c",
})
df["yield_tons_ha"] = df["yield_hg_ha"] / 10000  # hectograms/ha -> tonnes/ha
df = df.drop_duplicates()

print(f"Dataset shape after dropping duplicates: {df.shape}")
print(f"Missing values:\n{df.isnull().sum()}")
print(f"Crops tracked ({len(df['crop'].unique())}): {list(df['crop'].unique())}")
print(f"Countries tracked: {df['country'].nunique()}")
print(f"Year span: {df['year'].min()} to {df['year'].max()}")

# %% 2. Exploratory Data Analysis (EDA)
print("\n--- Generating EDA Figures ---")

# 2.1 Global yield trend over time
yearly = df.groupby("year")["yield_tons_ha"].mean().reset_index()
plt.figure(figsize=(7, 4))
plt.plot(yearly["year"], yearly["yield_tons_ha"], marker="o", color="#3B6B4A", linewidth=2.2, markersize=5)
plt.title("Global Average Crop Yield (1990–2013)", fontsize=13, fontweight="bold", pad=12)
plt.xlabel("Year", fontsize=11)
plt.ylabel("Yield (tonnes/ha)", fontsize=11)
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(FIG_DIR / "yield_trend.png", dpi=150)
plt.close()
print("Saved yield_trend.png")

# 2.2 Yield distribution by crop
crop_order = df.groupby("crop")["yield_tons_ha"].mean().sort_values(ascending=False).index
plt.figure(figsize=(9, 4.8))
sns.boxplot(data=df, x="crop", y="yield_tons_ha", order=crop_order, color="#C08A2E", fliersize=2)
plt.xticks(rotation=35, ha="right", fontsize=10)
plt.title("Yield Distribution by Crop Type", fontsize=13, fontweight="bold", pad=12)
plt.xlabel("Crop", fontsize=11)
plt.ylabel("Yield (tonnes/ha)", fontsize=11)
plt.tight_layout()
plt.savefig(FIG_DIR / "yield_by_crop.png", dpi=150)
plt.close()
print("Saved yield_by_crop.png")

# 2.3 Climate correlations
plt.figure(figsize=(6, 5))
corr = df[["yield_tons_ha", "rainfall_mm", "pesticides_tonnes", "avg_temp_c", "year"]].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn", center=0, cbar_kws={"shrink": 0.8})
plt.title("Correlation Matrix (Climate & Agricultural Features)", fontsize=12, fontweight="bold", pad=12)
plt.tight_layout()
plt.savefig(FIG_DIR / "correlation_heatmap.png", dpi=150)
plt.close()
print("Saved correlation_heatmap.png")

# 2.4 Temperature vs yield faceted by major crops
major_crops = ["Maize", "Wheat", "Rice, paddy", "Soybeans"]
palette = {"Maize": "#C08A2E", "Wheat": "#3B6B4A", "Rice, paddy": "#A6462B", "Soybeans": "#2A4E36"}
plt.figure(figsize=(8.5, 6.5))
for i, crop in enumerate(major_crops, 1):
    plt.subplot(2, 2, i)
    sub = df[df["crop"] == crop]
    plt.scatter(sub["avg_temp_c"], sub["yield_tons_ha"], alpha=0.35, s=12, color=palette[crop])
    plt.title(crop, fontweight="bold", fontsize=11)
    plt.xlabel("Avg Temp (°C)", fontsize=10)
    plt.ylabel("Yield (t/ha)", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.4)
plt.suptitle("Temperature vs Yield for Key Global Staples", fontsize=13, fontweight="bold", y=0.98)
plt.tight_layout()
plt.savefig(FIG_DIR / "temp_vs_yield_by_crop.png", dpi=150)
plt.close()
print("Saved temp_vs_yield_by_crop.png")

# %% 3. Feature Engineering & Temporal Train/Test Split
print("\n--- Feature Engineering & Temporal Split ---")
FEATURES_NUM = ["year", "rainfall_mm", "pesticides_tonnes", "avg_temp_c"]
FEATURES_CAT = ["country", "crop"]
TARGET = "yield_tons_ha"

train_df = df[df["year"] <= 2009].copy()
test_df = df[df["year"] > 2009].copy()
print(f"Temporal Split -> Train (<=2009): {len(train_df):,} rows | Test (2010-2013): {len(test_df):,} rows")

X_train, y_train = train_df[FEATURES_NUM + FEATURES_CAT], train_df[TARGET]
X_test, y_test = test_df[FEATURES_NUM + FEATURES_CAT], test_df[TARGET]

preprocess = ColumnTransformer([
    ("num", StandardScaler(), FEATURES_NUM),
    ("cat", OneHotEncoder(handle_unknown="ignore"), FEATURES_CAT),
])

# %% 4. Train & Compare Models
print("\n--- Training Benchmark Models ---")
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Random Forest": RandomForestRegressor(
        n_estimators=300, max_depth=None, min_samples_leaf=2,
        random_state=42, n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05, random_state=42
    ),
}

results = []
fitted = {}
for name, model in models.items():
    pipe = Pipeline([("prep", preprocess), ("model", model)])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    r2 = r2_score(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    results.append({"model": name, "r2": round(r2, 4), "rmse": round(rmse, 3), "mae": round(mae, 3)})
    fitted[name] = pipe
    print(f"  {name:20s} -> R² = {r2:.4f} | RMSE = {rmse:.3f} t/ha | MAE = {mae:.3f} t/ha")

results_df = pd.DataFrame(results).sort_values("r2", ascending=False)
results_df.to_csv(EXPORT_DIR / "model_comparison.csv", index=False)
print("\nModel Comparison Table:")
print(results_df.to_string(index=False))

best_name = results_df.iloc[0]["model"]
best_model = fitted[best_name]
print(f"\nBest Model Selected: {best_name}")

# %% 5. Feature Importance Analysis
print("\n--- Calculating Feature Importances ---")
rf_pipe = fitted["Random Forest"]
ohe_cols = rf_pipe.named_steps["prep"].named_transformers_["cat"].get_feature_names_out(FEATURES_CAT)
all_cols = FEATURES_NUM + list(ohe_cols)
importances = rf_pipe.named_steps["model"].feature_importances_

imp_df = pd.DataFrame({"feature": all_cols, "importance": importances})
imp_df["group"] = imp_df["feature"].apply(
    lambda f: "country" if f.startswith("country_") else ("crop" if f.startswith("crop_") else f)
)
grouped_imp = imp_df.groupby("group")["importance"].sum().sort_values(ascending=False)
print("Grouped Importances:\n", grouped_imp)

plt.figure(figsize=(6.5, 4))
grouped_imp.plot(kind="barh", color="#C08A2E", edgecolor="#212B1E", linewidth=0.5)
plt.gca().invert_yaxis()
plt.title("Feature Importance by Variable Group (Random Forest)", fontsize=11, fontweight="bold", pad=10)
plt.xlabel("Relative Importance", fontsize=10)
plt.tight_layout()
plt.savefig(FIG_DIR / "feature_importance.png", dpi=150)
plt.close()
print("Saved feature_importance.png")

# %% 6. Save Production Model
model_path = MODEL_DIR / "best_model.joblib"
joblib.dump(best_model, model_path)
print(f"\nSerialized production model saved to: {model_path}")

# %% 7. Export Lightweight Browser Model (In-Browser Tree Inference)
print("\n--- Training Lightweight Browser Model ---")
top_countries = df["country"].value_counts().head(20).index.tolist()
df["country_bucketed"] = df["country"].where(df["country"].isin(top_countries), "Other")

FEATURES_LITE_NUM = ["year", "rainfall_mm", "pesticides_tonnes", "avg_temp_c"]
FEATURES_LITE_CAT = ["crop", "country_bucketed"]

X_lite = df[FEATURES_LITE_NUM + FEATURES_LITE_CAT]
y_lite = df[TARGET]

lite_ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
X_lite_cat = lite_ohe.fit_transform(X_lite[FEATURES_LITE_CAT])
cat_names = lite_ohe.get_feature_names_out(FEATURES_LITE_CAT)
X_lite_full = np.hstack([X_lite[FEATURES_LITE_NUM].values, X_lite_cat])
feature_names_lite = FEATURES_LITE_NUM + list(cat_names)

lite_rf = RandomForestRegressor(n_estimators=25, max_depth=8, min_samples_leaf=5, random_state=42, n_jobs=-1)
lite_rf.fit(X_lite_full, y_lite)
lite_r2 = r2_score(y_lite, lite_rf.predict(X_lite_full))
print(f"Lightweight browser model (train R² in-sample): {lite_r2:.3f}")


def export_tree(tree):
    t = tree.tree_

    def node_to_dict(i):
        if t.children_left[i] == t.children_right[i] == -1:
            return {"leaf": round(float(t.value[i][0][0]), 4)}
        return {
            "f": int(t.feature[i]),
            "t": round(float(t.threshold[i]), 5),
            "l": node_to_dict(t.children_left[i]),
            "r": node_to_dict(t.children_right[i]),
        }

    return node_to_dict(0)


forest_json = {
    "feature_names": feature_names_lite,
    "num_features": FEATURES_LITE_NUM,
    "crop_options": sorted(df["crop"].unique().tolist()),
    "country_options": sorted(top_countries) + ["Other"],
    "trees": [export_tree(t) for t in lite_rf.estimators_],
    "feature_ranges": {
        "year": [1990, 2030],
        "rainfall_mm": [int(df["rainfall_mm"].quantile(0.02)), int(df["rainfall_mm"].quantile(0.98))],
        "pesticides_tonnes": [0, int(df["pesticides_tonnes"].quantile(0.95))],
        "avg_temp_c": [float(round(df["avg_temp_c"].min(), 2)), float(round(df["avg_temp_c"].max(), 2))],
    },
}

with open(EXPORT_DIR / "forest_model.json", "w", encoding="utf-8") as f:
    json.dump(forest_json, f, separators=(",", ":"))

print(f"Exported {len(forest_json['trees'])} trees to {EXPORT_DIR / 'forest_model.json'} ({len(feature_names_lite)} features)")

# %% 8. Export Aggregated Dashboard Data
print("\n--- Exporting Dashboard Data ---")
dashboard_data = {
    "yearly_global_yield": yearly.round(3).to_dict(orient="records"),
    "yield_by_crop": (
        df.groupby("crop")["yield_tons_ha"].agg(["mean", "median", "std"]).round(3)
        .reset_index().to_dict(orient="records")
    ),
    "top_countries_by_yield": (
        df.groupby("country")["yield_tons_ha"].mean().sort_values(ascending=False).head(12)
        .round(3).reset_index().to_dict(orient="records")
    ),
    "bottom_countries_by_yield": (
        df.groupby("country")["yield_tons_ha"].mean().sort_values().head(12)
        .round(3).reset_index().to_dict(orient="records")
    ),
    "temp_vs_yield_by_crop": {
        crop: df[df["crop"] == crop][["avg_temp_c", "yield_tons_ha"]]
        .sample(min(150, (df["crop"] == crop).sum()), random_state=1)
        .round(3).to_dict(orient="records")
        for crop in major_crops
    },
    "correlation": corr.round(3).to_dict(),
    "model_comparison": results_df.to_dict(orient="records"),
    "feature_importance": grouped_imp.round(4).reset_index().rename(columns={"index": "group"}).to_dict(orient="records"),
    "dataset_stats": {
        "rows": int(len(df)),
        "countries": int(df["country"].nunique()),
        "crops": int(df["crop"].nunique()),
        "year_min": int(df["year"].min()),
        "year_max": int(df["year"].max()),
    },
}

with open(EXPORT_DIR / "dashboard_data.json", "w", encoding="utf-8") as f:
    json.dump(dashboard_data, f, separators=(",", ":"))

print(f"Exported dashboard data to {EXPORT_DIR / 'dashboard_data.json'}")

# Also assemble index.html if dashboard_template.html exists
template_path = PROJECT_ROOT / "dashboard_template.html"
index_path = PROJECT_ROOT / "index.html"
if template_path.exists():
    print(f"\nCompiling {index_path.name} from template...")
    with open(template_path, "r", encoding="utf-8") as tf:
        html = tf.read()
    
    with open(EXPORT_DIR / "forest_model.json", "r", encoding="utf-8") as ff:
        forest_str = ff.read()
    with open(EXPORT_DIR / "dashboard_data.json", "r", encoding="utf-8") as df_file:
        dashboard_str = df_file.read()
    
    html = html.replace("__FOREST_JSON__", forest_str)
    html = html.replace("__DASHBOARD_JSON__", dashboard_str)
    
    with open(index_path, "w", encoding="utf-8") as out_f:
        out_f.write(html)
    print(f"Successfully created standalone {index_path.name} ({os.path.getsize(index_path):,} bytes)!")

print("\n--- Pipeline Execution Completed Successfully! ---")
