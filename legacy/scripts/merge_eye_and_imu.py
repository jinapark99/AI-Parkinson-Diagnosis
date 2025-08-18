import pandas as pd

# ──────────────────────────────
# 🔄 파일 경로
# ──────────────────────────────
eye_path = "eye_movement_coordinates.csv"
imu_path = "imu_data.csv"  # 필요 시 파일명 수정

# ──────────────────────────────
# 📂 CSV 로드
# ──────────────────────────────
eye_df = pd.read_csv(eye_path)
imu_df = pd.read_csv(imu_path)

# ──────────────────────────────
# 📌 Timestamp 정렬 및 병합 (nearest join)
# ──────────────────────────────
eye_df['timestamp'] = pd.to_datetime(eye_df['timestamp'], unit='s')
imu_df['timestamp'] = pd.to_datetime(imu_df['timestamp'], unit='s')

# 가장 가까운 timestamp 기준 병합 (pandas 1.5+ 필요)
merged_df = pd.merge_asof(
    eye_df.sort_values('timestamp'),
    imu_df.sort_values('timestamp'),
    on='timestamp',
    direction='nearest',
    tolerance=pd.Timedelta('50ms')  # 0.05초 차이까지 허용
)

# ──────────────────────────────
# 💾 저장
# ──────────────────────────────
merged_df.to_csv("merged_eye_imu.csv", index=False)
print("✅ 저장 완료: merged_eye_imu.csv")
