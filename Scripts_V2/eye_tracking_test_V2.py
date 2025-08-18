import cv2
import mediapipe as mp
import numpy as np
from screeninfo import get_monitors
import json
import os
from datetime import datetime
import time
import random
import csv
import pandas as pd

def get_iris_center(landmarks, iris_indices, w, h):
    """홍채 중심점 계산"""
    x_coords = [landmarks[idx].x * w for idx in iris_indices]
    y_coords = [landmarks[idx].y * h for idx in iris_indices]
    return int(np.mean(x_coords)), int(np.mean(y_coords))

def smooth_coordinates(coord_history, window_size=5):
    """좌표 스무딩 (이동평균 필터)"""
    if len(coord_history) < window_size:
        return coord_history[-1] if coord_history else (0, 0)
    
    recent_coords = coord_history[-window_size:]
    avg_x = int(np.mean([coord[0] for idx, coord in enumerate(recent_coords)]))
    avg_y = int(np.mean([coord[1] for idx, coord in enumerate(recent_coords)]))
    return avg_x, avg_y

def calculate_eye_movement_sync(left_eye_history, right_eye_history, threshold=10):
    """양쪽 눈의 움직임 동기화 정도 계산"""
    if len(left_eye_history) < 2 or len(right_eye_history) < 2:
        return 1.0
    
    # 최근 움직임 벡터 계산
    left_dx = left_eye_history[-1][0] - left_eye_history[-2][0]
    left_dy = left_eye_history[-1][1] - left_eye_history[-2][1]
    right_dx = right_eye_history[-1][0] - right_eye_history[-2][0]
    right_dy = right_eye_history[-1][1] - right_eye_history[-2][1]
    
    # 움직임 방향의 유사도 계산
    left_magnitude = np.sqrt(left_dx**2 + left_dy**2)
    right_magnitude = np.sqrt(right_dx**2 + right_dy**2)
    
    if left_magnitude < threshold and right_magnitude < threshold:
        return 1.0  # 둘 다 정지 상태
    
    # 움직임 방향의 코사인 유사도
    if left_magnitude > 0 and right_magnitude > 0:
        dot_product = left_dx * right_dx + left_dy * right_dy
        cos_similarity = dot_product / (left_magnitude * right_magnitude)
        return max(0, cos_similarity)
    
    return 0.0

def generate_random_target_position(screen_width, screen_height, margin=100, current_pos=None, used_positions=None):
    """화면 전체에서 랜덤한 목표 위치 생성 (긴 거리 보장, 중복 방지)"""
    if current_pos is None:
        # 첫 번째 위치는 완전 랜덤
        x = random.randint(margin, screen_width - margin)
        y = random.randint(margin, screen_height - margin)
        return x, y
    
    # 현재 위치와 충분히 멀리 떨어진 위치 생성
    current_x, current_y = current_pos
    attempts = 0
    max_attempts = 50  # 더 많은 시도 허용
    
    while attempts < max_attempts:
        # 화면 전체에서 랜덤 선택
        x = random.randint(margin, screen_width - margin)
        y = random.randint(margin, screen_height - margin)
        
        # 현재 위치와의 거리 계산
        distance = np.sqrt((x - current_x)**2 + (y - current_y)**2)
        
        # 화면 대각선 거리의 60-80% 이상 떨어져야 함
        screen_diagonal = np.sqrt(screen_width**2 + screen_height**2)
        min_distance = screen_diagonal * 0.6
        max_distance = screen_diagonal * 0.9
        
        # 사용된 위치와도 충분히 떨어져야 함
        too_close_to_used = False
        if used_positions:
            for used_pos in used_positions:
                used_distance = np.sqrt((x - used_pos[0])**2 + (y - used_pos[1])**2)
                if used_distance < min_distance * 0.5:  # 사용된 위치와 너무 가까우면 제외
                    too_close_to_used = True
                    break
        
        if min_distance <= distance <= max_distance and not too_close_to_used:
            return x, y
        
        attempts += 1
    
    # 최대 시도 횟수 초과 시 강제로 멀리 떨어진 위치 생성
    # 화면의 반대편 구역에서 선택
    if current_x < screen_width // 2:
        x = random.randint(screen_width - margin - 300, screen_width - margin)
    else:
        x = random.randint(margin, margin + 300)
    
    if current_y < screen_height // 2:
        y = random.randint(screen_height - margin - 300, screen_height - margin)
    else:
        y = random.randint(margin, margin + 300)
    
    return x, y

def run_random_eye_tracking_test():
    """랜덤 위치 빠른 움직임 아이트래킹 테스트"""
    print("🎯 랜덤 위치 빠른 움직임 아이트래킹 테스트 시작...")
    
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
    
    # 홍채 인덱스
    left_iris = [474, 475, 476, 477]
    right_iris = [469, 470, 471, 472]
    
    # 카메라 열기
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 카메라를 열 수 없습니다.")
        return
    
    print("✅ 카메라 연결 성공!")
    print("🎯 테스트 설명:")
    print("   🚀 빠른 움직임으로 반응 시간 측정")
    print("   🎲 랜덤 위치로 반복학습 방지")
    print("   📏 화면 전체 활용한 긴 거리 이동")
    print("   ⚡ 빠른 정지로 갑작스러운 방향 전환")
    print("⏰ 30초 후 자동 종료됩니다.")
    print("📱 'q' 키를 누르면 즉시 종료됩니다.")
    
    # OpenCV 창 생성 및 최대화
    cv2.namedWindow("랜덤 아이트래킹 테스트", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("랜덤 아이트래킹 테스트", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    # 테스트 시작 UI 표시
    print("🚀 테스트 시작 UI를 표시합니다...")
    start_countdown = 5
    while start_countdown > 0:
        # 시작 화면 생성
        start_screen = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
        
        # 제목
        cv2.putText(start_screen, "Random Eye Tracking Test", 
                    (screen_width//2 - 400, screen_height//2 - 150), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 3)
        
        # 카운트다운
        cv2.putText(start_screen, f"Starting test in {start_countdown} seconds...", 
                    (screen_width//2 - 300, screen_height//2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
        
        # 설명
        cv2.putText(start_screen, "Follow the red dot to measure reaction time", 
                    (screen_width//2 - 350, screen_height//2 + 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        cv2.imshow("랜덤 아이트래킹 테스트", start_screen)
        cv2.waitKey(1000)  # 1초 대기
        start_countdown -= 1
    
    # 테스트 상태 변수
    test_start_time = time.time()
    
    # 모든 목표점을 미리 생성하여 순서대로 사용
    all_targets = []
    for i in range(20):  # 20개의 목표점 미리 생성
        if i == 0:
            target = generate_random_target_position(screen_width, screen_height)
        else:
            # 이전 목표점과 충분히 멀리 떨어진 위치 생성
            prev_target = all_targets[-1]
            target = generate_random_target_position(screen_width, screen_height, current_pos=prev_target)
        all_targets.append(target)
    
    current_target = all_targets[0]
    next_target = all_targets[1]
    target_index = 1
    
    print(f"🎯 미리 생성된 목표점들:")
    for i, target in enumerate(all_targets[:5]):  # 처음 5개만 표시
        print(f"   {i}: {target}")
    
    # 움직임 상태
    is_moving = False
    movement_start_time = None
    movement_duration = random.uniform(1.5, 3.0)  # 1.5~3초간 이동
    stop_duration = random.uniform(0.8, 1.5)      # 0.8~1.5초간 정지
    
    # 눈 움직임 기록
    left_eye_history = []
    right_eye_history = []
    reaction_times = []
    eye_sync_scores = []
    
    # CSV 데이터 수집을 위한 변수들
    csv_data = []
    frame_count = 0
    
    # 테스트 결과
    test_results = {
        "total_movements": 0,
        "average_reaction_time": None,
        "eye_sync_scores": [],
        "movement_patterns": []
    }
    
    print(f"🎯 첫 번째 목표: {current_target}")
    print(f"🎯 다음 목표: {next_target}")
    
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
        
        # 현재 시간 계산
        current_time = time.time()
        elapsed_time = current_time - test_start_time
        
        # 움직임 상태 업데이트
        if not is_moving:
            # 정지 상태에서 다음 움직임 시작
            if movement_start_time is None:
                movement_start_time = current_time
            elif current_time - movement_start_time >= stop_duration:
                # 정지 시간 완료, 움직임 시작
                is_moving = True
                movement_start_time = current_time
                
                # 목표점 업데이트: 현재 위치를 이전 목표로, 다음 목표를 현재 목표로
                current_target = next_target
                
                # 새로운 다음 목표점 생성
                target_index += 1
                if target_index < len(all_targets):
                    next_target = all_targets[target_index]
                else:
                    # 모든 목표점을 사용했으면 새로운 목표점 생성
                    next_target = generate_random_target_position(screen_width, screen_height, current_pos=current_target)
                    all_targets.append(next_target)
                
                movement_duration = random.uniform(1.5, 3.0)
                stop_duration = random.uniform(0.8, 1.5)
                test_results["total_movements"] += 1
                
                print(f"🚀 {test_results['total_movements']}번째 움직임 시작!")
                print(f"🎯 이동 경로: {current_target} → {next_target}")
                print(f"⏱️ 이동 시간: {movement_duration:.1f}초, 정지 시간: {stop_duration:.1f}초")
                print(f"🔍 현재 목표: {current_target}")
                print(f"🔍 다음 목표: {next_target}")
        else:
            # 움직임 상태에서 이동 완료 확인
            if current_time - movement_start_time >= movement_duration:
                # 이동 완료, 정지 상태로 전환
                is_moving = False
                movement_start_time = current_time
                print(f"⏸️ {test_results['total_movements']}번째 움직임 완료, 정지 상태")
        
        # 목표점 위치 계산 (움직임 중일 때)
        if is_moving:
            progress = (current_time - movement_start_time) / movement_duration
            start_x, start_y = current_target
            end_x, end_y = next_target
            
            # 부드러운 이동 (ease-in-out)
            if progress < 0.5:
                # ease-in
                t = 2 * progress * progress
            else:
                # ease-out
                t = 1 - 2 * (1 - progress) * (1 - progress)
            
            target_x = int(start_x + (end_x - start_x) * t)
            target_y = int(start_y + (end_y - start_y) * t)
        else:
            target_x, target_y = current_target
        
        # 목표점 그리기
        if is_moving:
            # 움직임 중: 빨간 점
            cv2.circle(frame_fullscreen, (target_x, target_y), 20, (0, 0, 255), -1)
            cv2.circle(frame_fullscreen, (target_x, target_y), 25, (255, 255, 255), 4)
        else:
            # 정지 상태: 파란 점
            cv2.circle(frame_fullscreen, (target_x, target_y), 20, (255, 0, 0), -1)
            cv2.circle(frame_fullscreen, (target_x, target_y), 25, (255, 255, 255), 4)
        
        # 다음 목표점 표시 (작은 점으로)
        cv2.circle(frame_fullscreen, next_target, 8, (0, 255, 255), -1)
        cv2.circle(frame_fullscreen, next_target, 12, (255, 255, 255), 2)
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            h, w = frame.shape[:2]
            
            # 홍채 중심점 계산
            left_center = get_iris_center(landmarks, left_iris, w, h)
            right_center = get_iris_center(landmarks, right_iris, w, h)
            
            # 좌표계 변환: 원본 → 전체화면
            scale_x = screen_width / w
            scale_y = screen_height / h
            
            # 왼쪽 홍채를 전체화면에 표시
            left_screen_x = int(left_center[0] * scale_x)
            left_screen_y = int(left_center[1] * scale_y)
            
            # 오른쪽 홍채를 전체화면에 표시
            right_screen_x = int(right_center[0] * scale_x)
            right_screen_y = int(right_center[1] * scale_y)
            
            # 홍채 좌표 기록
            left_eye_history.append((left_screen_x, left_screen_y))
            right_eye_history.append((right_screen_x, right_screen_y))
            
            # 최근 30개 좌표만 유지
            if len(left_eye_history) > 30:
                left_eye_history.pop(0)
            if len(right_eye_history) > 30:
                right_eye_history.pop(0)
            
            # 스무딩 적용
            if len(left_eye_history) >= 5:
                left_smooth = smooth_coordinates(left_eye_history)
                right_smooth = smooth_coordinates(right_eye_history)
            else:
                left_smooth = (left_screen_x, left_screen_y)
                right_smooth = (right_screen_x, right_screen_y)
            
            # 홍채 표시 (스무딩된 좌표)
            cv2.circle(frame_fullscreen, left_smooth, 4, (0, 255, 0), -1)  # 녹색
            cv2.circle(frame_fullscreen, left_smooth, 6, (255, 255, 255), 1)  # 흰색 테두리
            cv2.circle(frame_fullscreen, right_smooth, 4, (0, 255, 0), -1)  # 녹색
            cv2.circle(frame_fullscreen, right_smooth, 6, (255, 255, 255), 1)  # 흰색 테두리
            
            # 동체시력 점수 계산
            eye_sync_score = calculate_eye_movement_sync(left_eye_history, right_eye_history)
            eye_sync_scores.append(eye_sync_score)
            
            # CSV 데이터 수집
            csv_row = {
                'frame': frame_count,
                'timestamp': elapsed_time,
                'left_eye_x': left_smooth[0],
                'left_eye_y': left_smooth[1],
                'right_eye_x': right_smooth[0],
                'right_eye_y': right_smooth[1],
                'target_x': target_x,
                'target_y': target_y,
                'is_moving': is_moving,
                'eye_sync_score': eye_sync_score,
                'movement_phase': test_results["total_movements"],
                'reaction_detected': False,
                'reaction_time': None
            }
            
            # 반응 시간이 감지된 경우 기록
            if is_moving and len(left_eye_history) >= 5 and len(right_eye_history) >= 5:
                left_movement = abs(left_smooth[0] - left_eye_history[-5][0]) + abs(left_smooth[1] - left_eye_history[-5][1])
                right_movement = abs(right_smooth[0] - right_eye_history[-5][0]) + abs(right_smooth[1] - right_eye_history[-5][1])
                
                if left_movement > 8 and right_movement > 8:
                    reaction_time = current_time - movement_start_time
                    if reaction_time < 0.5:
                        reaction_times.append(reaction_time)
                        csv_row['reaction_detected'] = True
                        csv_row['reaction_time'] = reaction_time
                        print(f"⚡ 반응 시간: {reaction_time:.3f}초")
            
            csv_data.append(csv_row)
            frame_count += 1
            

            
            # 시선 방향을 화살표로 표시
            arrow_color = (255, 0, 0) if is_moving else (0, 255, 0)  # 움직임 중: 파란색, 정지: 녹색
            arrow_thickness = 2
            
            # 왼쪽 홍채에서 목표점까지 화살표
            cv2.arrowedLine(frame_fullscreen, left_smooth, 
                           (target_x, target_y), arrow_color, arrow_thickness, tipLength=0.3)
            
            # 오른쪽 홍채에서 목표점까지 화살표
            cv2.arrowedLine(frame_fullscreen, right_smooth, 
                           (target_x, target_y), arrow_color, arrow_thickness, tipLength=0.3)
            
            # 정보 표시
            cv2.putText(frame_fullscreen, f"Left: {left_smooth}", 
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(frame_fullscreen, f"Right: {right_smooth}", 
                        (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(frame_fullscreen, f"Target: ({target_x}, {target_y})", 
                        (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255) if is_moving else (255, 0, 0), 2)
            cv2.putText(frame_fullscreen, f"Time: {elapsed_time:.1f}s", 
                        (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.putText(frame_fullscreen, f"Status: {'이동중' if is_moving else '정지'}", 
                        (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255) if is_moving else (255, 0, 0), 2)
            cv2.putText(frame_fullscreen, f"Movement: {test_results['total_movements']}", 
                        (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.putText(frame_fullscreen, f"Eye Sync: {eye_sync_score:.2f}", 
                        (50, 350), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
            
            if reaction_times:
                avg_reaction = np.mean(reaction_times)
                cv2.putText(frame_fullscreen, f"Avg Reaction: {avg_reaction:.3f}s", 
                            (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        else:
            cv2.putText(frame_fullscreen, "얼굴이 감지되지 않습니다", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
        
        # 전체화면으로 표시
        cv2.imshow("랜덤 아이트래킹 테스트", frame_fullscreen)
        
        # 30초 타이머 체크
        if elapsed_time >= 30:
            print("⏰ 30초 테스트 완료! 결과를 분석합니다...")
            
            # 테스트 완료 UI 표시
            print("✅ 테스트 완료 UI를 표시합니다...")
            completion_countdown = 3
            while completion_countdown > 0:
                # 완료 화면 생성
                completion_screen = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
                
                # 제목
                cv2.putText(completion_screen, "Test Complete!", 
                            (screen_width//2 - 200, screen_height//2 - 150), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 0), 4)
                
                # 카운트다운
                cv2.putText(completion_screen, f"Window closes in {completion_countdown} seconds", 
                            (screen_width//2 - 250, screen_height//2), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
                
                # 결과 미리보기
                if reaction_times:
                    avg_reaction = np.mean(reaction_times)
                    cv2.putText(completion_screen, f"Average Reaction Time: {avg_reaction:.3f}s", 
                                (screen_width//2 - 200, screen_height//2 + 100), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                
                cv2.imshow("랜덤 아이트래킹 테스트", completion_screen)
                cv2.waitKey(1000)  # 1초 대기
                completion_countdown -= 1
            
            break
        
        # 키 입력 처리
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("🛑 사용자가 테스트를 중단했습니다.")
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # 테스트 결과 출력
    print("\n📊 테스트 결과:")
    print(f"🔄 총 움직임 횟수: {test_results['total_movements']}")
    
    if reaction_times:
        avg_reaction = np.mean(reaction_times)
        min_reaction = min(reaction_times)
        max_reaction = max(reaction_times)
        print(f"⏱️ 평균 반응 시간: {avg_reaction:.3f}초")
        print(f"⏱️ 최소 반응 시간: {min_reaction:.3f}초")
        print(f"⏱️ 최대 반응 시간: {max_reaction:.3f}초")
        
        if avg_reaction < 0.2:
            print("✅ 반응 속도: 매우 빠름")
        elif avg_reaction < 0.3:
            print("✅ 반응 속도: 빠름")
        elif avg_reaction < 0.4:
            print("⚠️ 반응 속도: 보통")
        else:
            print("❌ 반응 속도: 느림")
    else:
        print("⏱️ 반응 시간: 측정 실패")
    
    if eye_sync_scores:
        avg_sync = np.mean(eye_sync_scores)
        print(f"👀 평균 동체시력 점수: {avg_sync:.3f} (1.0 = 완벽, 0.0 = 완전 분리)")
        
        if avg_sync > 0.8:
            print("✅ 동체시력: 정상")
        elif avg_sync > 0.6:
            print("⚠️ 동체시력: 경미한 이상")
        else:
            print("❌ 동체시력: 이상 가능성")
    
    # 결과를 파일로 저장
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Data_V2/eye_tracking_data 폴더에 저장
    data_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Data_V2", "eye_tracking_data")
    
    # 폴더가 없으면 생성
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
        print(f"📁 {data_folder} 폴더를 생성했습니다.")
    
    # 날짜별 하위 폴더 생성
    date_folder = os.path.join(data_folder, datetime.now().strftime("%Y-%m"))
    if not os.path.exists(date_folder):
        os.makedirs(date_folder)
        print(f"📁 {date_folder} 폴더를 생성했습니다.")
    
    result_file = os.path.join(date_folder, f"random_eye_tracking_result_{timestamp}.json")
    csv_file = os.path.join(date_folder, f"eye_tracking_data_{timestamp}.csv")
    
    # JSON 파일 저장용 데이터 준비
    save_data = {
        "timestamp": timestamp,
        "screen_resolution": f"{screen_width}x{screen_height}",
        "test_duration": elapsed_time,
        "total_movements": test_results["total_movements"],
        "reaction_times": reaction_times,
        "average_reaction_time": np.mean(reaction_times) if reaction_times else None,
        "min_reaction_time": min(reaction_times) if reaction_times else None,
        "max_reaction_time": max(reaction_times) if reaction_times else None,
        "average_eye_sync_score": np.mean(eye_sync_scores) if eye_sync_scores else None,
        "eye_sync_scores": eye_sync_scores,
        "total_frames": frame_count,
        "csv_file": csv_file
    }
    
    # JSON 파일 저장
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)
    
    # CSV 파일 저장
    if csv_data:
        df = pd.DataFrame(csv_data)
        df.to_csv(csv_file, index=False, encoding='utf-8')
        print(f"📊 CSV 데이터가 저장되었습니다:")
        print(f"   📁 경로: {csv_file}")
        print(f"   📊 총 {len(csv_data)}개 프레임 데이터")
        print(f"   ⏱️ {len([row for row in csv_data if row['reaction_detected']])}개 반응 시간 기록")
    
    print(f"💾 JSON 결과가 저장되었습니다:")
    print(f"   📁 경로: {result_file}")
    print(f"   📂 저장 위치: {date_folder}")
    print("✅ 랜덤 아이트래킹 테스트 완료!")

if __name__ == "__main__":
    run_random_eye_tracking_test()
