import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# Font settings (English only to avoid font warnings)
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['axes.unicode_minus'] = False

# ──────────────────────────────
# 📂 File Paths & Settings
# ──────────────────────────────
eye_path = "eye_movement_coordinates.csv"
target_path = "coordenadas_pelota.csv"
FPS = 60  # Adjust based on recording setup

# ──────────────────────────────
# 📥 Load CSVs
# ──────────────────────────────
eye_df = pd.read_csv(eye_path)
eye_df["frame_id"] = range(len(eye_df))  # Add frame index
target_df = pd.read_csv(target_path)

# ──────────────────────────────
# 🕒 Timestamp Calculation (frame_id / fps)
# ──────────────────────────────
eye_df["timestamp"] = pd.to_datetime(eye_df["frame_id"] / FPS, unit="s")
target_df["timestamp"] = pd.to_datetime(target_df["frame_id"] / FPS, unit="s")

# ──────────────────────────────
# 🔗 Merge on timestamp (nearest match)
# ──────────────────────────────
merged_df = pd.merge_asof(
    eye_df.sort_values("timestamp"),
    target_df.sort_values("timestamp"),
    on="timestamp",
    direction="nearest",
    tolerance=pd.Timedelta("50ms"),
    suffixes=("_eye", "_target")
)

# ──────────────────────────────
# 📏 Distance Error Calculation
# ──────────────────────────────
merged_df["distance"] = np.sqrt(
    (merged_df["center_x"] - merged_df["x"])**2 +
    (merged_df["center_y"] - merged_df["y"])**2
)

# ──────────────────────────────
# 📊 Print Statistics
# ──────────────────────────────
print("\n📊 Prediction Error Stats (MRL-based):")
print("Mean Error: {:.2f}px".format(merged_df["distance"].mean()))
print("Standard Deviation: {:.2f}px".format(merged_df["distance"].std()))
print("Max Error: {:.2f}px".format(merged_df["distance"].max()))

# ──────────────────────────────
# 📈 Plot Distance Error Over Time
# ──────────────────────────────
plt.figure(figsize=(12, 5))
plt.plot(merged_df["timestamp"], merged_df["distance"], label="Prediction Error (px)", color="crimson")
plt.xlabel("Time")
plt.ylabel("Distance Error (px)")
plt.title("Gaze vs Target Distance (MRL)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("distance_MRL_prediction.png")
plt.show()

# ──────────────────────────────
# 🧭 Plot X Trajectory Comparison
# ──────────────────────────────
plt.figure(figsize=(12, 5))
plt.plot(merged_df["timestamp"], merged_df["center_x"], label="Gaze X", color="blue")
plt.plot(merged_df["timestamp"], merged_df["x"], label="Target X", color="orange", linestyle="--")
plt.xlabel("Time")
plt.ylabel("Horizontal Position (px)")
plt.title("Gaze vs Target Trajectory (X-axis)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("trajectory_x_MRL.png")
plt.show()

# ──────────────────────────────
# 🧭 Plot Y Trajectory Comparison
# ──────────────────────────────
plt.figure(figsize=(12, 5))
plt.plot(merged_df["timestamp"], merged_df["center_y"], label="Gaze Y", color="green")
plt.plot(merged_df["timestamp"], merged_df["y"], label="Target Y", color="purple", linestyle="--")
plt.xlabel("Time")
plt.ylabel("Vertical Position (px)")
plt.title("Gaze vs Target Trajectory (Y-axis)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("trajectory_y_MRL.png")
plt.show()

# ──────────────────────────────
# 💾 Save Merged CSV
# ──────────────────────────────
merged_df.to_csv("merged_eye_target_MRL.csv", index=False)
print("✅ Merge and analysis complete → merged_eye_target_MRL.csv")
