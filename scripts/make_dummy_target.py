import pandas as pd

FPS = 30               # 너의 eye_analysis랑 동일하게
DURATION = 10          # 10초 기준 (원하면 바꿔도 됨)
frames = list(range(FPS * DURATION))

# 자극이 오른쪽으로 5픽셀씩 움직이고, Y는 고정
df = pd.DataFrame({
    "frame_id": frames,
    "x": [400 + i * 5 for i in frames],  # 시작점 400에서 매 프레임마다 5씩 증가
    "y": [350 for _ in frames]           # y는 고정
})

df.to_csv("coordenadas_pelota.csv", index=False)
print("✅ 자극 좌표 저장 완료: coordenadas_pelota.csv")
