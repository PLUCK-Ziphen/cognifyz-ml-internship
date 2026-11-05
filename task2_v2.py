# Task 2 - Restaurant Recommendation System (v2 - simple)
# Cognifyz Machine Learning Internship
# Author: Akhilesh Eerni
#
# Goal: given a user's preferred cuisine, city, and price range,
# recommend restaurants that match -- ranked by rating.

import pandas as pd

df = pd.read_csv("Dataset_for_cognifyz.csv")

# Only recommend restaurants that actually have a real rating.
df = df[df["Aggregate rating"] > 0]
df["Cuisines"] = df["Cuisines"].fillna("Unknown")


def recommend_restaurants(cuisine, city, price_range, top_n=5):
    """
    Very simple content-based filtering:
    1. Keep restaurants whose Cuisines column mentions the requested
       cuisine (case-insensitive text match).
    2. Keep restaurants in the requested city.
    3. Keep restaurants at the requested price range.
    4. Sort what's left by rating (then votes) and return the top N.
    """
    matches = df[
        df["Cuisines"].str.contains(cuisine, case=False)
        & (df["City"].str.lower() == city.lower())
        & (df["Price range"] == price_range)
    ]

    # If nothing matches that exactly, relax the price range so the
    # user still gets useful suggestions instead of an empty result.
    if matches.empty:
        print(f"No exact price-range match for {cuisine} in {city}. Showing all price ranges instead.\n")
        matches = df[
            df["Cuisines"].str.contains(cuisine, case=False)
            & (df["City"].str.lower() == city.lower())
        ]

    matches = matches.sort_values(["Aggregate rating", "Votes"], ascending=False)
    return matches.head(top_n)[
        ["Restaurant Name", "City", "Cuisines", "Average Cost for two",
         "Price range", "Aggregate rating", "Votes"]
    ]


# ---------------------------------------------------------
# Test it with a few sample user preferences
# ---------------------------------------------------------
test_users = [
    {"cuisine": "North Indian", "city": "New Delhi", "price_range": 2},
    {"cuisine": "Italian", "city": "Gurgaon", "price_range": 3},
    {"cuisine": "Cafe", "city": "Noida", "price_range": 1},
]

for user in test_users:
    print("=" * 70)
    print(f"Looking for: {user['cuisine']} food in {user['city']}, price range {user['price_range']}")
    recommendations = recommend_restaurants(
        cuisine=user["cuisine"], city=user["city"], price_range=user["price_range"]
    )
    print(recommendations.to_string(index=False))
    print()
