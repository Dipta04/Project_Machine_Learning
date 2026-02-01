
import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# =====================
# Load dataset
# =====================
df = pd.read_csv("bangladesh_student_performance_2018.csv")

print(df)

# Drop date column
if 'date' in df.columns:
    df.drop(columns=['date'], inplace=True)

# Target and features
X = df.drop('hsc_result', axis=1)
y = df['hsc_result']

# =====================
# Column split
# =====================
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

# =====================
# Preprocessing
# =====================
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', num_transformer, numeric_features),
    ('cat', cat_transformer, categorical_features)
])

# =====================
# Random Forest Model
# =====================
rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    min_samples_split=2,
    random_state=42,
    n_jobs=-1
)

# =====================
# Full Pipeline
# =====================
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', rf_model)
])

# =====================
# Train-test split
# ====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


rf_pipeline.fit(X_train, y_train)

# =====================
# Evaluation
# =====================
y_pred = rf_pipeline.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.4f}")
print(f"R2 Score: {r2:.4f}")

# =====================
# Save model (IMPORTANT)
# =====================

with open("student_rf_pipeline.pkl", "wb") as f:
    pickle.dump(rf_pipeline, f)

print("✅ Random Forest pipeline saved as student_rf_pipeline.pkl")