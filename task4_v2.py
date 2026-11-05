# Task 4 - Location-based Restaurant Analysis (v2 - simple)
# Cognifyz Machine Learning Internship
# Author: Akhilesh Eerni
#
# Goal: explore where restaurants are located, and see how location
# relates to things like restaurant count and average rating.
# (This task is about analysis/visualization, not prediction -- so
# there's no model to train here, just grouping and charts.)

import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. Load the data
# ---------------------------------------------------------
df = pd.read_csv("Dataset_for_cognifyz.csv")

# A rating of 0 means "Not rated", not actually a rating of zero. We
# keep those restaurants on the location map (they're still real
# restaurants) but leave them out of anything that averages ratings,
# so the averages aren't dragged down by restaurants with no reviews.
rated_df = df[df["Aggregate rating"] > 0]

# ---------------------------------------------------------
# 2. Plot every restaurant by location
# ---------------------------------------------------------
# Latitude/Longitude naturally trace out a map shape even without a
# real basemap underneath -- clusters of dots show where restaurants
# are concentrated.
plt.figure(figsize=(10, 6))
scatter = plt.scatter(
    df["Longitude"], df["Latitude"],
    s=5, alpha=0.4, c=df["Aggregate rating"], cmap="RdYlGn"
)
plt.colorbar(scatter, label="Aggregate rating (0 = not rated)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Restaurant Locations (colored by rating)")
plt.tight_layout()
plt.savefig("restaurant_locations.png")
print("Saved restaurant_locations.png")

# ---------------------------------------------------------
# 3. Group restaurants by city
# ---------------------------------------------------------
city_stats = rated_df.groupby("City").agg(
    restaurant_count=("Restaurant Name", "count"),
    avg_rating=("Aggregate rating", "mean"),
    avg_cost=("Average Cost for two", "mean"),
).sort_values("restaurant_count", ascending=False)

print("\nTop 10 cities by number of restaurants:")
print(city_stats.head(10).round(2))

# ---------------------------------------------------------
# 4. Which cities have the best-rated restaurants?
# ---------------------------------------------------------
# Only look at cities with at least 10 restaurants -- otherwise a
# city with just 1 or 2 lucky restaurants could top the list by
# chance, which wouldn't be a meaningful comparison.
established_cities = city_stats[city_stats["restaurant_count"] >= 10]
top_rated_cities = established_cities.sort_values("avg_rating", ascending=False)

print("\nTop 10 highest-rated cities (min. 10 restaurants):")
print(top_rated_cities.head(10).round(2))

# ---------------------------------------------------------
# 5. Simple side-by-side comparison charts
# ---------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

top10_count = city_stats.head(10)
axes[0].barh(top10_count.index[::-1], top10_count["restaurant_count"][::-1], color="#4C72B0")
axes[0].set_xlabel("Number of restaurants")
axes[0].set_title("Top 10 Cities by Restaurant Count")

top10_rating = top_rated_cities.head(10)
axes[1].barh(top10_rating.index[::-1], top10_rating["avg_rating"][::-1], color="#55A868")
axes[1].set_xlabel("Average rating")
axes[1].set_title("Top 10 Highest-Rated Cities (min. 10 restaurants)")

plt.tight_layout()
plt.savefig("city_comparison.png")
print("\nSaved city_comparison.png")

# ---------------------------------------------------------
# 6. A couple of plain-language insights
# ---------------------------------------------------------
busiest_city = city_stats.index[0]
best_rated_city = top_rated_cities.index[0]

print(
    f"\nInsight: '{busiest_city}' has the most restaurants "
    f"({city_stats.iloc[0]['restaurant_count']:.0f}), but its average "
    f"rating is only {city_stats.iloc[0]['avg_rating']:.2f}."
)
print(
    f"Insight: '{best_rated_city}' has the highest average rating among "
    f"well-established cities ({top_rated_cities.iloc[0]['avg_rating']:.2f}), "
    f"with {top_rated_cities.iloc[0]['restaurant_count']:.0f} restaurants -- "
    f"more restaurants doesn't automatically mean better ones."
)
