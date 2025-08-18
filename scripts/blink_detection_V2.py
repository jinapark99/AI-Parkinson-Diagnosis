import cv2
import mediapipe as mp
import numpy as np
from screeninfo import get_monitors
import time
import csv
from datetime import datetime

def get_eye_aspect_ratio(landmarks, eye_indices):
    """눈 종횡비(EAR: Eye Aspect Ratio) 계산 - MediaPipe 표준 방식"""
    # 눈의 수직 거리들 (윗꺼풀과 아랫꺼풀 사이)
    A = np.linalg.norm(np.array([landmarks[eye_indices[1]].x, landmarks[eye_indices[1]].y]) - 
                       np.array([landmarks[eye_indices[5]].x, landmarks[eye_indices[5]].y]))
    B = np.linalg.norm(np.array([landmarks[eye_indices[2]].x, landmarks[eye_indices[2]].y]) - 
                       np.array([landmarks[eye_indices[4]].x, landmarks[eye_indices[4]].y]))
    
    # 눈의 수평 거리 (안쪽 모서리와 바깥쪽 모서리 사이)
    C = np.linalg.norm(np.array([landmarks[eye_indices[0]].x, landmarks[eye_indices[0]].y]) - 
                       np.array([landmarks[eye_indices[3]].x, landmarks[eye_indices[3]].y]))
    
    # EAR 계산 (0으로 나누기 방지)
    if C > 0:
        ear = (A + B) / (2.0 * C)
    else:
        ear = 0.0
    
    return ear

def get_ear_for_mediapipe_face_mesh(landmarks, eye_indices):
    """MediaPipe Face Mesh용 EAR 계산 - 더 정확한 눈 랜드마크 사용"""
    # 눈의 수직 거리들 (윗꺼풀과 아랫꺼풀 사이)
    A = np.linalg.norm(np.array([landmarks[eye_indices[1]].x, landmarks[eye_indices[1]].y]) - 
                       np.array([landmarks[eye_indices[5]].x, landmarks[eye_indices[5]].y]))
    B = np.linalg.norm(np.array([landmarks[eye_indices[2]].x, landmarks[eye_indices[2]].y]) - 
                       np.array([landmarks[eye_indices[4]].x, landmarks[eye_indices[4]].y]))
    
    # 눈의 수평 거리 (안쪽 모서리와 바깥쪽 모서리 사이)
    C = np.linalg.norm(np.array([landmarks[eye_indices[0]].x, landmarks[eye_indices[0]].y]) - 
                       np.array([landmarks[eye_indices[3]].x, landmarks[eye_indices[3]].y]))
    
    # EAR 계산 (0으로 나누기 방지)
    if C > 0:
        ear = (A + B) / (2.0 * C)
    else:
        ear = 0.0
    
    return ear

def detect_blinks():
    """30초 동안 눈 깜빡임 측정"""
    print("👁️ 눈 깜빡임 측정을 시작합니다...")
    print("📱 30초 동안 자연스럽게 눈을 깜빡이세요.")
    print("🔄 'r' 키를 누르면 다시 측정할 수 있습니다.")
    print("❌ 'q' 키를 누르면 언제든지 종료됩니다.")
    print("⏰ 30초 측정 완료 시 자동으로 결과가 저장되고 프로그램이 종료됩니다.")
    
    # 모니터 정보 가져오기
    monitor = get_monitors()[0]
    screen_width = monitor.width
    screen_height = monitor.height
    print(f"📺 모니터 크기: {screen_width} x {screen_height}")
    
    # MediaPipe 초기화
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # EAR 계산용 눈 랜드마크 인덱스 (MediaPipe Face Mesh - EAR 최적화)
    # 왼쪽 눈: [안쪽 모서리, 윗꺼풀1, 윗꺼풀2, 바깥쪽 모서리, 아랫꺼풀1, 아랫꺼풀2]
    left_eye_ear = [33, 160, 158, 133, 153, 144]  # 왼쪽 눈 EAR용
    # 오른쪽 눈: [안쪽 모서리, 윗꺼풀1, 윗꺼풀2, 바깥쪽 모서리, 아랫꺼풀1, 아랫꺼풀2]
    right_eye_ear = [362, 385, 387, 263, 373, 380]  # 오른쪽 눈 EAR용
    
    # 기존 방식용 (디버그 표시용)
    left_eye = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
    right_eye = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    
    # 카메라 열기
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 카메라를 열 수 없습니다.")
        return
    
    print("✅ 카메라 연결 성공!")
    
    # 눈 깜빡임 관련 변수 (EAR 기반)
    blink_counter = 0
    baseline_left_ear = None        # 스페이스바 누를 때의 기준 EAR
    baseline_right_ear = None       # 스페이스바 누를 때의 기준 EAR
    left_ear_threshold = None       # 왼쪽 눈 EAR 임계값
    right_ear_threshold = None      # 오른쪽 눈 EAR 임계값
    min_blink_interval = 0.3        # 최소 깜빡임 간격 (초)
    
    # 눈 상태 추적
    eyes_closed = False
    last_blink_time = 0
    
    # 측정 관련 변수
    start_time = time.time()
    test_duration = 30  # 30초
    is_measuring = False
    
    # 디버그용 변수
    frame_count = 0
    
    # OpenCV 창 생성 및 최대화
    cv2.namedWindow("Blink Detection", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Blink Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # 프레임 뒤집기
        frame = cv2.flip(frame, 1)
        
        # 홍채 찾기
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        
        # 프레임을 전체화면 크기로 리사이즈
        frame_fullscreen = cv2.resize(frame, (screen_width, screen_height))
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            h, w = frame.shape[:2]
            
            # EAR 계산
            left_ear = get_ear_for_mediapipe_face_mesh(landmarks, left_eye_ear)
            right_ear = get_ear_for_mediapipe_face_mesh(landmarks, right_eye_ear)
            
            # 눈 깜빡임 감지 (EAR 기반)
            current_time = time.time()
            
            # 기준 EAR가 설정되었을 때만 깜빡임 감지
            if baseline_left_ear is not None and baseline_right_ear is not None:
                # EAR가 임계값 이하이면 눈이 감긴 것으로 판단 (EAR는 눈이 감기면 작아짐)
                if left_ear < left_ear_threshold and right_ear < right_ear_threshold:
                    if not eyes_closed:  # 이전에 눈이 열려있었다면
                        if current_time - last_blink_time > min_blink_interval:
                            blink_counter += 1
                            last_blink_time = current_time
                            print(f"👁️ Blink detected! Left EAR: {left_ear:.4f}/{left_ear_threshold:.4f}, Right EAR: {right_ear:.4f}/{right_ear_threshold:.4f}")
                            print(f"📊 Total blinks: {blink_counter}")
                        eyes_closed = True
                else:
                    # EAR가 임계값보다 크면 눈이 열린 것으로 판단
                    eyes_closed = False
            else:
                # 기준 EAR가 설정되지 않았으면 기본값 사용
                eyes_closed = False
            
            # 디버그 출력
            frame_count += 1
            if frame_count % 3 == 0:  # 더 자주 출력
                if baseline_left_ear is not None:
                    print(f"Frame {frame_count}: Left EAR {left_ear:.4f}/{left_ear_threshold:.4f}, Right EAR {right_ear:.4f}/{right_ear_threshold:.4f}")
                    print(f"Status: {'CLOSED' if eyes_closed else 'OPEN'}, Blinks: {blink_counter}")
                else:
                    print(f"Frame {frame_count}: Baseline not set. Press SPACEBAR to start.")
                    print(f"Current EARs: Left {left_ear:.4f}, Right {right_ear:.4f}")
            
            # 측정 시작/종료 처리
            if not is_measuring:
                # 측정 시작 버튼 표시 (영어로)
                cv2.putText(frame_fullscreen, "Blink Detection Start", (screen_width//2 - 200, screen_height//2 - 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                cv2.putText(frame_fullscreen, "Press SPACEBAR to start 30-second test", (screen_width//2 - 300, screen_height//2), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                cv2.putText(frame_fullscreen, "Blink naturally during the test", (screen_width//2 - 250, screen_height//2 + 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            else:
                # 측정 중
                elapsed_time = current_time - start_time
                remaining_time = max(0, test_duration - elapsed_time)
                
                if remaining_time > 0:
                    # 남은 시간 표시
                    cv2.putText(frame_fullscreen, f"Testing... {remaining_time:.1f}s left", 
                                (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                    
                    # 깜빡임 횟수 표시
                    cv2.putText(frame_fullscreen, f"Blinks: {blink_counter}", 
                                (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 3)
                    
                    # 현재 EAR 표시
                    if 'left_ear' in locals() and 'right_ear' in locals():
                        cv2.putText(frame_fullscreen, f"Left EAR: {left_ear:.4f}", 
                                    (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                        cv2.putText(frame_fullscreen, f"Right EAR: {right_ear:.4f}", 
                                    (50, 230), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                        
                        # 기준 EAR 정보 표시
                        if baseline_left_ear is not None:
                            cv2.putText(frame_fullscreen, f"Left Baseline EAR: {baseline_left_ear:.4f}", 
                                        (50, 260), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                            cv2.putText(frame_fullscreen, f"Right Baseline EAR: {baseline_right_ear:.4f}", 
                                        (50, 290), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                            cv2.putText(frame_fullscreen, f"Left EAR Threshold: {left_ear_threshold:.4f}", 
                                        (50, 320), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                            cv2.putText(frame_fullscreen, f"Right EAR Threshold: {right_ear_threshold:.4f}", 
                                        (50, 350), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                        else:
                            cv2.putText(frame_fullscreen, "Press SPACEBAR to set baseline", 
                                        (50, 260), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                        
                        # 눈 상태 표시
                        if eyes_closed:
                            eye_status = "BLINKING"
                            eye_color = (0, 0, 255)  # 빨간색
                        else:
                            eye_status = "NORMAL"
                            eye_color = (0, 255, 0)  # 초록색
                        
                        cv2.putText(frame_fullscreen, f"Status: {eye_status}", 
                                    (50, 380), cv2.FONT_HERSHEY_SIMPLEX, 1.0, eye_color, 2)
                else:
                    # 측정 완료
                    is_measuring = False
                    
                    # 결과 표시 (3초간 표시 후 자동 종료)
                    cv2.putText(frame_fullscreen, "Test Complete!", (screen_width//2 - 150, screen_height//2 - 100), 
                                cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 3)
                    cv2.putText(frame_fullscreen, f"Blinks in 30s: {blink_counter}", 
                                (screen_width//2 - 200, screen_height//2), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 3)
                    cv2.putText(frame_fullscreen, f"Blinks per minute: {blink_counter * 2}", 
                                (screen_width//2 - 200, screen_height//2 + 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 3)
                    cv2.putText(frame_fullscreen, "Results will be saved automatically", 
                                (screen_width//2 - 250, screen_height//2 + 100), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                    cv2.putText(frame_fullscreen, "Program will close in 3 seconds...", 
                                (screen_width//2 - 250, screen_height//2 + 150), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                    
                    # 결과 자동 저장
                    save_results(blink_counter)
                    
                    # 3초 대기 후 종료
                    cv2.waitKey(3000)
                    break
            
            # 눈 랜드마크 표시 (디버그용) - 전체 눈 영역 표시
            # 왼쪽 눈 전체 표시
            for idx in left_eye:
                x = int(landmarks[idx].x * w * screen_width / w)
                y = int(landmarks[idx].y * h * screen_height / h)
                cv2.circle(frame_fullscreen, (x, y), 2, (0, 255, 255), -1)  # 노란색
            
            # 오른쪽 눈 전체 표시
            for idx in right_eye:
                x = int(landmarks[idx].x * w * screen_width / w)
                y = int(landmarks[idx].y * h * screen_height / h)
                cv2.circle(frame_fullscreen, (x, y), 2, (255, 0, 255), -1)  # 마젠타색
            
            # 눈 윤곽선 표시 (선으로 연결)
            # 왼쪽 눈 윤곽선
            left_eye_points = []
            for idx in left_eye:
                x = int(landmarks[idx].x * w * screen_width / w)
                y = int(landmarks[idx].y * h * screen_height / h)
                left_eye_points.append([x, y])
            
            if len(left_eye_points) > 0:
                left_eye_points = np.array(left_eye_points, np.int32)
                cv2.polylines(frame_fullscreen, [left_eye_points], True, (0, 255, 0), 1)  # 초록색 윤곽선
            
            # 오른쪽 눈 윤곽선
            right_eye_points = []
            for idx in right_eye:
                x = int(landmarks[idx].x * w * screen_width / w)
                y = int(landmarks[idx].y * h * screen_height / h)
                right_eye_points.append([x, y])
            
            if len(right_eye_points) > 0:
                right_eye_points = np.array(right_eye_points, np.int32)
                cv2.polylines(frame_fullscreen, [right_eye_points], True, (255, 0, 0), 1)  # 파란색 윤곽선
        
        else:
            cv2.putText(frame_fullscreen, "Face not detected", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
        
        # 전체화면으로 표시
        cv2.imshow("Blink Detection", frame_fullscreen)
        
        # 프레임 처리 속도 조절 (천천히 처리)
        cv2.waitKey(50)  # 50ms 대기 (20 FPS)
        
        # 키 입력 처리
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord(' ') and not is_measuring:  # 스페이스바
            print("🚀 눈 깜빡임 측정을 시작합니다!")
            
            # 현재 EAR를 기준으로 설정
            if 'left_ear' in locals() and 'right_ear' in locals():
                baseline_left_ear = left_ear
                baseline_right_ear = right_ear
                
                # EAR 임계값 설정 (EAR는 눈이 감기면 작아지므로 기준 EAR의 80%를 임계값으로)
                left_ear_threshold = baseline_left_ear * 0.8
                right_ear_threshold = baseline_right_ear * 0.8
                
                print(f"Baseline set - Left EAR: {baseline_left_ear:.4f} → Threshold: {left_ear_threshold:.4f}")
                print(f"Baseline set - Right EAR: {baseline_right_ear:.4f} → Threshold: {right_ear_threshold:.4f}")
                print("이제 눈을 깜빡이세요!")
            else:
                print("❌ 눈 랜드마크를 찾을 수 없습니다. 얼굴을 카메라에 맞춰주세요.")
                continue
            
            is_measuring = True
            start_time = time.time()
            blink_counter = 0
            last_blink_time = 0
            eyes_closed = False
        elif key == ord('r') and not is_measuring:  # 다시 측정
            print("🔄 측정을 다시 시작합니다!")
            is_measuring = False
            blink_counter = 0
            last_blink_time = 0
    
    cap.release()
    cv2.destroyAllWindows()
    print("✅ 눈 깜빡임 측정 종료")

def save_results(blink_count):
    """결과를 CSV 파일로 저장"""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"blink_test_{timestamp}.csv"
    
    # 분당 깜빡임 계산
    blinks_per_minute = blink_count * 2  # 30초 * 2 = 1분
    
    # 파킨슨병 판정 (참고용)
    if blinks_per_minute < 10:
        status = "낮음 (파킨슨병 의심)"
    elif blinks_per_minute < 15:
        status = "보통 (주의 필요)"
    else:
        status = "정상"
    
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['날짜', '시간', '30초_깜빡임_횟수', '분당_깜빡임_횟수', '상태', '참고사항']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        writer.writerow({
            '날짜': datetime.now().strftime("%Y-%m-%d"),
            '시간': datetime.now().strftime("%H:%M:%S"),
            '30초_깜빡임_횟수': blink_count,
            '분당_깜빡임_횟수': blinks_per_minute,
            '상태': status,
            '참고사항': '정상: 15-20회/분, 파킨슨병: 5-10회/분'
        })
    
    print(f"💾 결과가 {filename}에 저장되었습니다!")
    print(f"📊 30초 깜빡임: {blink_count}회, 분당: {blinks_per_minute}회")
    print(f"🔍 상태: {status}")

if __name__ == "__main__":
    detect_blinks()
