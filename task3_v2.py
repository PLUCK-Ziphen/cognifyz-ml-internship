# Task 3 - Cuisine Classification (v2 - simple)
# Cognifyz Machine Learning Internship
# Author: Akhilesh Eerni
#
# Goal: predict a restaurant's main cuisine from its other details
# (city, cost, votes, rating, etc.) using a classification model.

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ---------------------------------------------------------
# 1. Load the data
# ---------------------------------------------------------
df = pd.read_csv("Dataset_for_cognifyz.csv")

# Only restaurants with a real rating (0 means "Not rated", not "rated zero")
df = df[df["Aggregate rating"] > 0].copy()
df["Cuisines"] = df["Cuisines"].fillna("Unknown")

# ---------------------------------------------------------
# 2. Build the target: a restaurant's "main" cuisine
# ---------------------------------------------------------
# The Cuisines column often lists several cuisines per restaurant
# (e.g. "North Indian, Chinese"). To keep this a straightforward
# classification problem, we take just the first one listed as the
# restaurant's main cuisine.
df["Main Cuisine"] = df["Cuisines"].apply(lambda c: c.split(",")[0].strip())

# Some cuisines only appear a handful of times, which isn't enough
# data for the model to learn a pattern from. We keep only the 10
# most common main cuisines and drop the rest -- this covers about
# 73% of all rated restaurants, which is plenty to work with.
top_cuisines = df["Main Cuisine"].value_counts().head(10).index
df = df[df["Main Cuisine"].isin(top_cuisines)]

print("Cuisines being classified:", list(top_cuisines))
print("Restaurants remaining after filtering:", len(df))

# ---------------------------------------------------------
# 3. Pick the columns we'll use to predict cuisine
# ---------------------------------------------------------
features = [
    "City",
    "Average Cost for two",
    "Price range",
    "Votes",
    "Aggregate rating",
    "Has Table booking",
    "Has Online delivery",
]
target = "Main Cuisine"

X = df[features].copy()
y = df[target]

# ---------------------------------------------------------
# 4. Encode text columns into numbers
# ---------------------------------------------------------
for col in ["City", "Has Table booking", "Has Online delivery"]:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

# ---------------------------------------------------------
# 5. Train/test split
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------------------------------------
# 6. Train the model
# ---------------------------------------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ---------------------------------------------------------
# 7. Evaluate
# ---------------------------------------------------------
y_pred = model.predict(X_test)

print("\nAccuracy:", round(accuracy_score(y_test, y_pred), 3))
print("\nDetailed performance per cuisine:")
print(classification_report(y_test, y_pred))

# ---------------------------------------------------------
# 8. Look for challenges / bias in performance
# ---------------------------------------------------------
# North Indian makes up a much larger share of the data than any other
# cuisine, so the model gets far more examples to learn it from. That
# usually shows up above as much higher precision/recall for
# "North Indian" than for smaller classes like "Mithai" or
# "Street Food" -- the model simply hasn't seen enough of those to
# learn them as well. This class imbalance is the main limitation of
# this simple approach.
print("Restaurant count per cuisine used in training:")
print(df["Main Cuisine"].value_counts())

# For context: how well would we do by just always guessing the most
# common cuisine (North Indian) for every restaurant? If our model
# doesn't clearly beat this, that's an honest sign that City/Cost/
# Votes/Rating alone aren't strong enough clues to tell cuisines apart
# -- a real limitation worth stating rather than hiding.
most_common = y_train.mode()[0]
baseline_accuracy = (y_test == most_common).mean()
print(f"\nNaive baseline (always guess '{most_common}'): {baseline_accuracy:.3f} accuracy")
print(f"Our model's accuracy:                          {accuracy_score(y_test, y_pred):.3f}")
