import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ──────────────────────────────
# 📂 파일 경로 및 설정
# ──────────────────────────────
eye_path = "red_dot_coordinates_final_extrapoladas.csv"
target_path = "coordenadas_pelota.csv"
FPS = 60  # 프레임 속도 맞춰야 함

# ──────────────────────────────
# 📥 CSV 불러오기
# ──────────────────────────────
eye_df = pd.read_csv(eye_path)
target_df = pd.read_csv(target_path)

# ──────────────────────────────
# 🕒 timestamp 계산 (frame_id / fps)
# ──────────────────────────────
eye_df["timestamp"] = pd.to_datetime(eye_df["frame_id"] / FPS, unit="s")
target_df["timestamp"] = pd.to_datetime(target_df["frame_id"] / FPS, unit="s")

# ──────────────────────────────
# 🔗 병합 (timestamp 기준, 가까운 값으로)
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
# 📏 거리 계산
# ──────────────────────────────
merged_df["distance"] = np.sqrt(
    (merged_df["x_eye"] - merged_df["x_target"])**2 +
    (merged_df["y_eye"] - merged_df["y_target"])**2
)

# ──────────────────────────────
# 📊 통계 출력
# ──────────────────────────────
print("\n📊 예측 오차 통계 (MRL 기반):")
print("평균 오차: {:.2f}px".format(merged_df["distance"].mean()))
print("표준편차: {:.2f}px".format(merged_df["distance"].std()))
print("최대 오차: {:.2f}px".format(merged_df["distance"].max()))

# ──────────────────────────────
# 📈 거리 오차 시각화
# ──────────────────────────────
plt.figure(figsize=(12, 5))
plt.plot(merged_df["timestamp"], merged_df["distance"], label="오차 거리 (px)", color="crimson")
plt.xlabel("시간")
plt.ylabel("거리 오차 (픽셀)")
plt.title("시선 예측 vs 자극 거리 오차 (MRL 기반)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("distance_MRL_prediction.png")
plt.show()

# ──────────────────────────────
# 🧭 수평/수직 궤적 비교
# ──────────────────────────────
plt.figure(figsize=(12, 5))
plt.plot(merged_df["timestamp"], merged_df["x_eye"], label="시선 x", color="blue")
plt.plot(merged_df["timestamp"], merged_df["x_target"], label="자극 x", color="orange", linestyle="--")
plt.xlabel("시간")
plt.ylabel("수평 위치(px)")
plt.title("시선 vs 자극 궤적 (X)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("trajectory_x_MRL.png")
plt.show()

plt.figure(figsize=(12, 5))
plt.plot(merged_df["timestamp"], merged_df["y_eye"], label="시선 y", color="green")
plt.plot(merged_df["timestamp"], merged_df["y_target"], label="자극 y", color="purple", linestyle="--")
plt.xlabel("시간")
plt.ylabel("수직 위치(px)")
plt.title("시선 vs 자극 궤적 (Y)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("trajectory_y_MRL.png")
plt.show()

# ──────────────────────────────
# 💾 저장
# ──────────────────────────────
merged_df.to_csv("merged_eye_target_MRL.csv", index=False)
print("✅ 병합 및 분석 완료! → merged_eye_target_MRL.csv")
