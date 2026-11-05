# Task 1 - Restaurant Rating Prediction
# Cognifyz Machine Learning Internship
# Author: Akhilesh Eerni
#
# Goal: predict a restaurant's Aggregate rating using its other details
# (city, cuisine, cost, votes, etc.) with a regression model.

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. Load the data
# ---------------------------------------------------------
df = pd.read_csv("Dataset_for_cognifyz.csv")
print("Rows, Columns:", df.shape)

# ---------------------------------------------------------
# 2. Clean the data
# ---------------------------------------------------------
# A rating of 0 actually means "Not rated" (no reviews yet), not a real
# score. We drop those rows so the model only learns from real ratings.
df = df[df["Aggregate rating"] > 0]

# A few rows have no cuisine listed. Fill those with "Unknown" instead
# of dropping them.
df["Cuisines"] = df["Cuisines"].fillna("Unknown")

# ---------------------------------------------------------
# 3. Pick the columns we'll actually use
# ---------------------------------------------------------
# We drop ID/name/address type columns (they don't generalize), and we
# drop Rating text / Rating color because those are just word/color
# versions of the Aggregate rating itself -- using them would basically
# be giving the model the answer.
features = [
    "City",
    "Cuisines",
    "Average Cost for two",
    "Votes",
    "Price range",
    "Has Table booking",
    "Has Online delivery",
]
target = "Aggregate rating"

X = df[features].copy()
y = df[target]

# ---------------------------------------------------------
# 4. Encode text columns into numbers
# ---------------------------------------------------------
# Machine learning models need numbers, not text. LabelEncoder just
# assigns each unique category (e.g. each city name) an integer ID.
for col in ["City", "Cuisines", "Has Table booking", "Has Online delivery"]:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

# ---------------------------------------------------------
# 5. Train/test split
# ---------------------------------------------------------
# We hold out 20% of the data to test the model on restaurants it has
# never seen, so we get an honest measure of how well it works.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------------------------------------
# 6. Train the model
# ---------------------------------------------------------
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ---------------------------------------------------------
# 7. Evaluate
# ---------------------------------------------------------
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel performance on test data:")
print(f"Mean Squared Error : {mse:.3f}")
print(f"R-squared score     : {r2:.3f}")

# ---------------------------------------------------------
# 8. Which features mattered most?
# ---------------------------------------------------------
importances = pd.Series(model.feature_importances_, index=features)
importances = importances.sort_values(ascending=False)

print("\nFeature importance:")
print(importances)

plt.figure(figsize=(8, 5))
importances.plot(kind="barh")
plt.gca().invert_yaxis()
plt.xlabel("Importance")
plt.title("What Influences Restaurant Ratings Most?")
plt.tight_layout()
plt.savefig("feature_importance_simple.png")
print("\nSaved chart as feature_importance_simple.png")
