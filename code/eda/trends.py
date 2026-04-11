import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

df = pd.read_csv("tutorial_data/prices_round_0_day_-2.csv", sep=";")

# Create a global time axis (day * 10000 + timestamp is standard in Prosperity)
df["time"] = df["day"] * 10000 + df["timestamp"]

# Split by product
emeralds = df[df["product"] == "EMERALDS"].copy().reset_index(drop=True)
tomatoes = df[df["product"] == "TOMATOES"].copy().reset_index(drop=True)

window = 200  # lookback for "equilibrium"
tomatoes["rolling_mean"] = tomatoes["mid_price"].rolling(window).mean()
tomatoes["rolling_std"] = tomatoes["mid_price"].rolling(window).std()
tomatoes["z_score"] = (tomatoes["mid_price"] - tomatoes["rolling_mean"]) / tomatoes[
    "rolling_std"
]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

ax1.plot(tomatoes["time"], tomatoes["mid_price"], label="mid price")
ax1.plot(
    tomatoes["time"], tomatoes["rolling_mean"], label="rolling mean", linestyle="--"
)
ax1.legend()
ax1.set_title("Tomatoes Price vs Rolling Mean")

ax2.plot(tomatoes["time"], tomatoes["z_score"], color="purple")
ax2.axhline(0, color="black", linewidth=1)
ax2.axhline(2, color="red", linestyle="--", label="+2σ (sell signal)")
ax2.axhline(-2, color="green", linestyle="--", label="-2σ (buy signal)")
ax2.set_title("Z-Score: how far price is from 'normal'")
ax2.legend()

plt.tight_layout()
plt.show()

from statsmodels.tsa.stattools import adfuller

result = adfuller(tomatoes["mid_price"].dropna())
print(f"ADF Statistic : {result[0]:.4f}")
print(f"p-value       : {result[1]:.4f}")
print("→ MEAN REVERTING" if result[1] < 0.05 else "→ RANDOM WALK / TRENDING")
