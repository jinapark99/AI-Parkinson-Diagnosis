import cv2
import mediapipe as mp
import pandas as pd
import time
import argparse

# ──────────────────────────────
# 🎛️ 설정: argparse

# ──────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--visualize", action="store_true", help="Display webcam visualization")
parser.add_argument("--duration", type=int, default=30, help="Recording duration (seconds)")
args = parser.parse_args()

# ──────────────────────────────
# 🎯 MediaPipe 초기화
# ──────────────────────────────
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

# ──────────────────────────────
# 📌 랜드마크 인덱스
# ──────────────────────────────
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]
HEAD_POSITION = [1, 33, 263, 61, 291]

# ──────────────────────────────
# 🔧 함수 정의
# ──────────────────────────────

def get_avg_pixel_coords(landmarks, indices, width, height):
    """주어진 인덱스의 평균 좌표(pixel 기준)를 반환"""
    x = sum(landmarks.landmark[i].x for i in indices) / len(indices)
    y = sum(landmarks.landmark[i].y for i in indices) / len(indices)
    return int(x * width), int(y * height)

def compute_features(left, right, head):
    """중앙점, 좌우 거리, 눈-머리 상대 위치 등 추가 분석 지표 계산"""
    cx = (left[0] + right[0]) // 2
    cy = (left[1] + right[1]) // 2
    dx = abs(left[0] - right[0])
    dy = abs(left[1] - right[1])
    head_dx = abs(head[0] - cx)
    head_dy = abs(head[1] - cy)
    return [cx, cy, dx, dy, head_dx, head_dy]

def visualize_frame(frame, left, right, head, center):
    """시각화: 눈동자, 머리, 중앙점 표시"""
    cv2.circle(frame, left, 3, (0, 255, 0), -1)
    cv2.circle(frame, right, 3, (0, 255, 0), -1)
    cv2.circle(frame, head, 3, (255, 0, 0), -1)
    cv2.circle(frame, center, 2, (0, 0, 255), -1)
    return frame

# ──────────────────────────────
# 🎥 웹캠 캡처 설정
# ──────────────────────────────
cap = cv2.VideoCapture(0)
fps = 30
cap.set(cv2.CAP_PROP_FPS, fps)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# ──────────────────────────────
# 🔁 루프 시작
# ──────────────────────────────
eye_data = []
start_time = time.time()

while cap.isOpened() and (time.time() - start_time < args.duration):
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        lm = results.multi_face_landmarks[0]

        left = get_avg_pixel_coords(lm, LEFT_IRIS, width, height)
        right = get_avg_pixel_coords(lm, RIGHT_IRIS, width, height)
        head = get_avg_pixel_coords(lm, HEAD_POSITION, width, height)

        features = compute_features(left, right, head)
        timestamp = time.time()

        # 저장
        eye_data.append([
            timestamp,
            *left, *right, *head,
            *features
        ])

        # 시각화
        if args.visualize:
            frame = visualize_frame(frame, left, right, head, (features[0], features[1]))
            cv2.imshow("Eye Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

cap.release()
cv2.destroyAllWindows()

# ──────────────────────────────
# 💾 CSV 저장
# ──────────────────────────────
df = pd.DataFrame(eye_data, columns=[
    "timestamp",
    "left_x", "left_y",
    "right_x", "right_y",
    "head_x", "head_y",
    "center_x", "center_y",
    "eye_dx", "eye_dy",
    "head_eye_dx", "head_eye_dy"
])
df.to_csv("eye_movement_coordinates.csv", index=False)
print("✅ 저장 완료: eye_movement_coordinates.csv")
