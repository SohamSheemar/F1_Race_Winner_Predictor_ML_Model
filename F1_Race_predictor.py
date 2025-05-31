#import pre-requisites
import fastf1
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

#fastf1 caching- storing required data locally; optimizes speed
import os
os.makedirs("f1_cache", exist_ok=True)
fastf1.Cache.enable_cache("f1_cache")

#Load 2024 Monaco GP race data
session_2024 = fastf1.get_session(2024, 'Monaco', 'R') #R-> Race session
session_2024.load()
#includes SectorTime for more precise predictions
laps_2024 = session_2024.laps[["Driver", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]].copy()
laps_2024.dropna(inplace=True)

for col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
    laps_2024[f"{col} (s)"] = laps_2024[col].dt.total_seconds()

#sector avg per Driver
sector_times_2024 = laps_2024.groupby("Driver").agg({
    "Sector1Time (s)": "mean",
    "Sector2Time (s)": "mean",
    "Sector3Time (s)": "mean"
}).reset_index()

sector_times_2024["TotalSectorTime (s)"] = (
    sector_times_2024["Sector1Time (s)"] +
    sector_times_2024["Sector2Time (s)"] +
    sector_times_2024["Sector3Time (s)"]
)

#manually input Quali time for 2025 Monaco GP 
qualifying_2025 = pd.DataFrame({
    "Driver": ["NOR", "LEC", "PIA", "HAM", "VER", "HAD", "ALO", "OCO", "LAW", "ALB", "SAI", "TSU", "HUL", "RUS", "ANT", "BOR", "BEA", "GAS", "STR", "COL"],
    "QualifyingTime (s)": [69.954, 70.063, 70.129, 70.382, 70.669, 70.923, 70.924, 70.942, 71.129, 71.213, 71.362, 71.415, 71.596, 71.507, 71.880, 71.902, 71.979, 71.994, 72.563, 72.597]
})

#average lap times for 2025 season of drivers
average_2025 = {
   "SAI": 89.348,   "HAM": 89.379,   "LEC": 89.431,   "RUS": 89.545,   "VER": 89.566,
   "ALB": 89.650,   "ANT": 89.784,   "PIA": 89.940,   "GAS": 90.040,   "STR": 90.229,
   "LAW": 90.252,   "DOO": 90.368,   "TSU": 90.497,   "HAD": 90.675,   "ALO": 90.700,
   "OCO": 90.748,   "NOR": 90.751,   "BOR": 90.876,   "HUL": 90.917,   "BEA": 90.938
}

#Driver wet performance factor
driver_wet_performance = {
    "VER": 0.975, "HAM": 0.976, "LEC": 0.974, "NOR": 0.978, "ALO": 0.972,
    "RUS": 0.969, "SAI": 0.977, "TSU": 0.996, "OCO": 0.982, "GAS": 0.979,
    "STR": 0.980, "PIA": 0.975, "HUL": 0.981
}
qualifying_2025["WetPerformanceFactor"] = qualifying_2025["Driver"].map(driver_wet_performance)

#Fetch weather condition for the circuit
import requests
API_KEY = "your API KEY"
lat,lon = 43.73472, 7.42056

url = f"http://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
weather_data = requests.get(url).json()

forecast_time = "2025-05-25 18:30:00"
forecast = next((item for item in weather_data["list"] if item["dt_txt"] == forecast_time), None)

rain_prob = forecast["pop"] if forecast else 0
temperature = forecast["main"]["temp"] if forecast else 25

# only take into account wet performance if chance is greater than 75% for rain
if rain_prob >= 0.75:
    qualifying_2025["QualifyingTime"] = qualifying_2025["QualifyingTime (s)"] * qualifying_2025["WetPerformanceFactor"]
else:
    qualifying_2025["QualifyingTime"] = qualifying_2025["QualifyingTime (s)"]


#Merge Features
data = qualifying_2025.merge(sector_times_2024, on="Driver", how="left")
data["RainProbability"] = rain_prob
data["Temperature"] = temperature

#constructors points
team_points = {
    "McLaren": 279, "Mercedes": 147, "Red Bull": 131, "Williams": 51, "Ferrari": 114,
    "Haas": 20, "Aston Martin": 14, "Kick Sauber": 6, "Racing Bulls": 10, "Alpine": 7
}
max_points = max(team_points.values())
team_performance_score = {team: points / max_points for team, points in team_points.items()}

driver_to_team = {
   "NOR": "McLaren",      "LEC": "Ferrari",       "PIA": "McLaren",      "HAM": "Ferrari",       "VER": "Red Bull",
   "HAD": "Racing Bulls", "ALO": "Aston Martin",  "OCO": "Haas",         "LAW": "Racing Bulls",  "ALB": "Williams",
   "SAI": "Williams",     "TSU": "Red Bull",      "HUL": "Kick Sauber",  "RUS": "Mercedes",      "ANT": "Mercedes",
   "BOT": "Kick Sauber",  "BER": "Haas",          "GAS": "Alpine",       "STR": "Aston Martin",  "STO": "Alpine"
}

qualifying_2025["Team"] = qualifying_2025["Driver"].map(driver_to_team)
qualifying_2025["TeamPerformanceScore"] = qualifying_2025["Team"].map(team_performance_score)
qualifying_2025["Average2025Performance"] = qualifying_2025["Driver"].map(average_2025)

merged_data = qualifying_2025.merge(sector_times_2024[["Driver", "TotalSectorTime (s)"]], on="Driver", how="left")
merged_data["RainProbability"] = rain_prob
merged_data["Temperature"] = temperature
last_year_winner = "LEC" 
merged_data["LastYearWinner"] = (merged_data["Driver"] == last_year_winner).astype(int)

merged_data["QualifyingTime"] = merged_data["QualifyingTime"] ** 2

# Define features
X = merged_data[[
    "QualifyingTime", "RainProbability", "Temperature", "TeamPerformanceScore", "TotalSectorTime (s)", "Average2025Performance"
]].fillna(0)
y = laps_2024.groupby("Driver")["LapTime (s)"].mean().reindex(merged_data["Driver"])

clean_data = merged_data.copy()
clean_data["LapTime (s)"] = y.values
clean_data = clean_data.dropna(subset=["LapTime (s)"])

X = clean_data[[
    "QualifyingTime", "TeamPerformanceScore", "RainProbability", "Temperature", "TotalSectorTime (s)", "Average2025Performance"
]]
y = clean_data["LapTime (s)"]

#train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=39)
model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05,  max_depth=5, random_state=39)
model.fit(X_train, y_train)
clean_data["PredictedRaceTime (s)"] = model.predict(X)

final_results = clean_data.sort_values("PredictedRaceTime (s)")
print("Predicted 2025 Monaco GP Winner:")
print(final_results[["Driver", "PredictedRaceTime (s)"]])

# MAE
y_pred = model.predict(X_test)
print(f"Model Error (MAE): {mean_absolute_error(y_test, y_pred):.2f} seconds")
print(merged_data)

# compute residuals for each driver
clean_data["Residual"] = clean_data["LapTime (s)"] - clean_data["PredictedRaceTime (s)"]
driver_errors = clean_data.groupby("Driver")["Residual"].mean().sort_values()
print(driver_errors)

# plot effects of the team performance score
plt.figure(figsize=(12, 8))
plt.scatter(final_results["TeamPerformanceScore"],
            final_results["PredictedRaceTime (s)"],
            c=final_results["QualifyingTime"])

for i, team_performance_score in enumerate(final_results["Driver"]):
    plt.annotate(team_performance_score, (final_results["TeamPerformanceScore"].iloc[i], final_results["PredictedRaceTime (s)"].iloc[i]),
                 xytext=(5, 5), textcoords='offset points')

plt.colorbar(label="Qualifying Time")
plt.xlabel("Team Performance Score")
plt.ylabel("Predicted Race Time (s)")
plt.title("Effect of Team Performance on Predicted Race Results")
plt.tight_layout()
plt.savefig('team_performance_effect.png')
plt.show()

# plot how important each feature is in the model
feature_importance = model.feature_importances_
features = X.columns

plt.figure(figsize=(8, 6))
plt.barh(features, feature_importance, color='skyblue')
plt.xlabel("Importance")
plt.title("Feature Importance in Race Time Prediction")
plt.tight_layout()
plt.show()

# Check correlation between features and target
corr_matrix = clean_data[[
    "QualifyingTime", "RainProbability", "Temperature", "TeamPerformanceScore", 
    "LastYearWinner", "Average2025Performance", "TotalSectorTime (s)", "LapTime (s)"
]].corr()

print(corr_matrix)

