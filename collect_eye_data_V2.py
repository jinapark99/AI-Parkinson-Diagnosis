# collect_eye_data_V2.py
import os, time, csv, random
import numpy as np
import cv2
import mediapipe as mp
from datetime import datetime
from PIL import ImageFont, ImageDraw, Image
import screeninfo
from collections import deque


import threading

# ==== 캡처/절차 타이밍 ====
HOLD_AFTER_ALIGN_SEC = 2.0     # 얼굴 정렬 후 유지 시간
NOTICE_SEC = 1.5               # "기준 촬영 시작" 문구 유지 시간
COUNTDOWN_SEC = 3              # 3,2,1
RECORD_SEC = 30                # 촬영 시간
FLASH_SEC = 0.35               # 플래시 연출 시간

# ==== 카메라 해상도 (웹캠 고정) ====
CAP_W, CAP_H = 640, 480  # 웹캠이 이 해상도로 동작하면 가장 안정적


# ---------- Mediapipe & 랜드마크 ----------
mp_face = mp.solutions.face_mesh
R_OUT, R_IN = 33, 133
L_OUT, L_IN = 263, 362

# ---------- 픽셀 보정(카메라 → 모니터) ----------
CAMERA_PIXEL_MM  = 0.005    # 640x480, 1/4" 센서 추정 (5um)
MONITOR_PIXEL_MM = 0.1117   # MacBook Air M1 13.3" 가로 2560px 기준
CAM_TO_MONITOR_SCALE = CAMERA_PIXEL_MM / MONITOR_PIXEL_MM  # ≈ 0.04473

# ---------- 한글 폰트 ----------
def load_korean_font(size=36):
    candidates = [
        "/System/Library/Fonts/Supplemental/AppleGothic.ttf",     # macOS
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",             # macOS
        "C:/Windows/Fonts/malgun.ttf",                            # Windows
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"         # Linux
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return None

def put_korean_center(img_bgr, text, size=36, color=(255,255,255)):
    font = load_korean_font(size)
    if font is not None:
        img_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        try:
            # 최신 Pillow
            bbox = draw.textbbox((0, 0), text, font=font)
            tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
        except Exception:
            # 구버전 fallback
            tw, th = draw.textsize(text, font=font)
        w, h = img_bgr.shape[1], img_bgr.shape[0]
        pos = ((w - tw)//2, (h - th)//2)
        draw.text(pos, text, font=font, fill=tuple(int(c) for c in color))
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    # 폰트 실패 시 OpenCV 기본(영문)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    w, h = img_bgr.shape[1], img_bgr.shape[0]
    pos = ((w - tw)//2, (h + th)//2)
    cv2.putText(img_bgr, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
    return img_bgr

# ---------- 유틸 ----------
def estimate_lag_and_gain(sig, ref, fs_est=30):
    """sig(무차원 u/v) → ref(자극 px) 맞추는 지연(초)과 이득, 바이어스 추정"""
    n = min(len(sig), len(ref))
    if n < 20:
        return 0.0, 1.0, 0.0
    a = np.array(sig[:n]) - np.mean(sig[:n])
    b = np.array(ref[:n]) - np.mean(ref[:n])
    corr = np.correlate(a, b, mode='full')
    lag_samples = np.argmax(corr) - (n - 1)
    lag_seconds = lag_samples / fs_est

    # 지연 적용해 공통 구간 정렬
    if lag_samples > 0:
        a_cut, b_cut = a[lag_samples:], b[:len(a)-lag_samples]
    elif lag_samples < 0:
        a_cut, b_cut = a[:len(a)+lag_samples], b[-lag_samples:]
    else:
        a_cut, b_cut = a, b

    if len(a_cut) < 10:
        return 0.0, 1.0, 0.0

    A = np.vstack([a_cut, np.ones_like(a_cut)]).T
    gain, bias = np.linalg.lstsq(A, b_cut, rcond=None)[0]
    return float(lag_seconds), float(gain), float(bias)

def get_delayed(buf, delay_sec):
    """deque([(t,val),...])에서 (현재-지연) 시점과 가장 가까운 val 반환"""
    if not buf:
        return 0.0
    t_now = time.time()
    tgt = t_now - max(0.0, delay_sec)
    best = buf[0][1]
    for t_i, v in reversed(buf):
        if t_i <= tgt:
            best = v; break
    return float(best)

def get_xy(lms, idx, w, h):
    p = lms[idx]
    return p.x * w, p.y * h

def compute_metrics(lms, w, h):
    rx, ry   = get_xy(lms, R_OUT, w, h); rix, riy = get_xy(lms, R_IN, w, h)
    lx, ly   = get_xy(lms, L_OUT, w, h); lix, liy = get_xy(lms, L_IN, w, h)
    io = ((lx - rx)**2 + (ly - ry)**2)**0.5
    rcx, rcy = (rx + rix)/2, (ry + riy)/2
    lcx, lcy = (lx + lix)/2, (ly + liy)/2
    fcx, fcy = (rx + lx)/2, (ry + ly)/2
    return dict(inter_ocular_px=io,
                right_cx=rcx, right_cy=rcy,
                left_cx=lcx,  left_cy=lcy,
                face_cx=fcx,  face_cy=fcy)

def show_countdown_fast(cap, window, ox, oy, w, h, mon_w, mon_h, seconds=3):
    """Pillow/한글 없이 OpenCV 숫자만 사용해서 초경량 카운트다운(버벅임 최소화)."""
    for num in range(seconds, 0, -1):
        t_end = time.perf_counter() + 1.0  # 각 숫자를 1초 간 보여줌
        while time.perf_counter() < t_end:
            ret, frame = cap.read()
            if not ret: return False
            frame = cv2.flip(frame, 1)

            # 전체화면 캔버스에 중앙 배치
            canvas = np.zeros((mon_h, mon_w, 3), dtype=np.uint8)
            canvas[oy:oy+h, ox:ox+w] = frame

            # 숫자는 OpenCV 기본 폰트(빠름)
            text = str(num)
            scale, thickness = 3.5, 8
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
            x = ox + (w - tw)//2
            y = oy + (h + th)//2
            cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 255), thickness, cv2.LINE_AA)

            cv2.imshow(window, canvas)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return False
    return True


def shutter_flash(win, canvas, ms=150):
    white = np.full_like(canvas, 255)
    cv2.imshow(win, white); cv2.waitKey(ms)
    cv2.imshow(win, canvas); cv2.waitKey(max(1, ms//2))

def non_blocking_notice(cap, window, text, sec, overlay=None, font_size=40, color=(255,255,255)):
    """sec 동안 카메라 프레임을 계속 갱신하면서 안내문구를 표시(버벅임 줄임)."""
    t_end = time.perf_counter() + sec
    while time.perf_counter() < t_end:
        ret, frm = cap.read()
        if not ret: break
        frm = cv2.flip(frm, 1)

        # 전체화면 캔버스 중앙 배치
        h, w = frm.shape[:2]
        canvas = np.zeros((overlay["MON_H"], overlay["MON_W"], 3), dtype=np.uint8)
        canvas[overlay["oy"]:overlay["oy"]+h, overlay["ox"]:overlay["ox"]+w] = frm

        canvas = put_korean_center(canvas, text, font_size, color)
        cv2.imshow(window, canvas)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            return False
    return True


def face_inside_ellipse(x, y, cx, cy, a, b):
    # (x-cx)^2/a^2 + (y-cy)^2/b^2 <= 1
    return ((x-cx)**2)/(a*a) + ((y-cy)**2)/(b*b) <= 1.0

# ---------- 부드러운 랜덤 곡선(느린 속도) ----------
class SmoothRandomPath:
    def __init__(self, w, h, y_max_ratio=2/3):
        self.w, self.h = w, h
        # 시작 시 한 번만 난수 고정
        self.fx = random.uniform(0.05, 0.09)   # 속도(느리게)
        self.fy = random.uniform(0.05, 0.09)
        self.phx = random.uniform(0, 2*np.pi)  # 위상
        self.phy = random.uniform(0, 2*np.pi)
        self.ax  = w * 0.35                    # 진폭
        self.ay  = h * 0.25
        self.cx, self.cy = w//2, h//2
        self.y_limit = h * y_max_ratio

    def pos(self, t):
        x = int(self.cx + self.ax * np.sin(self.fx * t + self.phx))
        y_raw = self.cy + self.ay * np.sin(self.fy * t + self.phy)
        y = int(min(y_raw, self.y_limit))
        return x, y

# ---------- 메인 ----------
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--duration", type=float, default=30.0)  # 촬영 30초
    ap.add_argument("--outdir", default="data")
    ap.add_argument("--subject", default="test_user")
    args = ap.parse_args()

    # 모니터 해상도 & 전체화면
    mon = screeninfo.get_monitors()[0]
    MON_W, MON_H = mon.width, mon.height
    WIN = "Eye Tracking"
    cv2.namedWindow(WIN, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # 카메라
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # 출력 경로
    os.makedirs(args.outdir, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    csv_path = os.path.join(args.outdir, f"{args.subject}_{ts}.csv")

    with mp_face.FaceMesh(max_num_faces=1,
                          refine_landmarks=True,  # ← 아이트래킹 반응 보려면 켜두자
                          min_detection_confidence=0.5,
                          min_tracking_confidence=0.5) as fm, \
            open(csv_path, "w", newline="") as fcsv:

        writer = csv.writer(fcsv)
        writer.writerow([
            "timestamp",
            "stim_x", "stim_y",
            "eye_x_px", "eye_y_px"
        ])

        # ===== 1) 얼굴 맞추기(타원 가이드) =====
        baseline = None
        align_ok_at = None  # 타원 안에 들어온 뒤 머문 시작 시각

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            # 전체화면 캔버스(카메라 프레임 중앙 배치)
            canvas = np.zeros((MON_H, MON_W, 3), dtype=np.uint8)
            ox, oy = (MON_W - w) // 2, (MON_H - h) // 2  # 원점 오프셋
            canvas[oy:oy + h, ox:ox + w] = frame

            # 카메라 프레임 중앙 기준 안내 타원
            ecx, ecy = ox + w // 2, oy + h // 2
            a, b = int(w * 0.28), int(h * 0.35)
            cv2.ellipse(canvas, (ecx, ecy), (a, b), 0, 0, 360, (0, 255, 0), 2)
            canvas = put_korean_center(canvas, "화면에 얼굴을 맞춰주세요", 40, (0, 255, 0))

            # 얼굴이 타원 안에 들어와 2초(HOLD_AFTER_ALIGN_SEC) 유지되면 통과
            res = fm.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            inside = False
            if res.multi_face_landmarks:
                lms = res.multi_face_landmarks[0].landmark
                rx, ry = get_xy(lms, R_OUT, w, h)
                lx, ly = get_xy(lms, L_OUT, w, h)
                fcx, fcy = (rx + lx) / 2 + ox, (ry + ly) / 2 + oy  # 캔버스 좌표
                inside = face_inside_ellipse(fcx, fcy, ecx, ecy, a, b)

            if inside and align_ok_at is None:
                align_ok_at = time.perf_counter()
            if not inside:
                align_ok_at = None

            if align_ok_at and (time.perf_counter() - align_ok_at >= HOLD_AFTER_ALIGN_SEC):
                # 2초 유지 통과 → 다음 단계로
                break

            cv2.imshow(WIN, canvas)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return

        # ===== 2) 기준 촬영 안내 → 카운트다운 → 플래시 → baseline =====
        ret, frame = cap.read()
        if not ret:
            cap.release(); cv2.destroyAllWindows(); return
        frame = cv2.flip(frame, 1)
        canvas = np.zeros((MON_H, MON_W, 3), dtype=np.uint8)
        canvas[oy:oy+h, ox:ox+w] = frame
        # --- 기준 촬영 안내: NOTICE_SEC 동안 프레임 갱신하며 표시 ---
        t_end = time.perf_counter() + NOTICE_SEC
        while time.perf_counter() < t_end:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            canvas = np.zeros((MON_H, MON_W, 3), dtype=np.uint8)
            canvas[oy:oy + h, ox:ox + w] = frame
            canvas = put_korean_center(canvas, "기준 촬영 시작", 40, (255, 255, 255))
            cv2.imshow(WIN, canvas)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # 초경량 카운트다운(버벅임 최소화)
        ok = show_countdown_fast(cap, WIN, ox, oy, w, h, MON_W, MON_H, seconds=3)
        if not ok:
            cap.release();
            cv2.destroyAllWindows();
            return

        # 플래시 & baseline 측정
        ret, frame = cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            res = fm.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if res.multi_face_landmarks:
                m = compute_metrics(res.multi_face_landmarks[0].landmark, w, h)
                baseline = m["inter_ocular_px"]
            # 플래시 효과
            canvas = np.zeros((MON_H, MON_W, 3), dtype=np.uint8)
            canvas[oy:oy+h, ox:ox+w] = frame
            shutter_flash(WIN, canvas, ms=160)

        if baseline is None:
            # 최후수단: 다음 프레임에서 잡히면 사용
            baseline = 1.0


        # ===== 3) 시선 추적 안내(짧게) =====
        for _ in range(60):  # 약 2초
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            canvas = np.zeros((MON_H, MON_W, 3), dtype=np.uint8)
            canvas[oy:oy+h, ox:ox+w] = frame
            canvas = put_korean_center(canvas, "시선 추적 시작, 고개는 고정하세요", 36, (0,0,255))
            cv2.imshow(WIN, canvas); cv2.waitKey(33)

        # ===== 4) 빨간점 30초(부드러운 랜덤 곡선, 상단 2/3 제한) =====
        t0 = time.time()
        while time.time() - t0 < 30.0:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)

            canvas = np.zeros((MON_H, MON_W, 3), dtype=np.uint8)
            canvas[oy:oy + h, ox:ox + w] = frame

            # 부드러운 경로에서 점 위치
            SPEED = 9
            t = time.time() - t0
            stim_x, stim_y = path.pos(t * SPEED)

            roi = canvas[oy:oy + h, ox:ox + w]
            cv2.circle(roi, (int(stim_x), int(stim_y)), 12, (0, 0, 255), -1)

            # 얼굴/눈
            res = fm.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if res.multi_face_landmarks:
                m = compute_metrics(res.multi_face_landmarks[0].landmark, w, h)
                io = max(1e-6, m["inter_ocular_px"])
                cx = (m["right_cx"] + m["left_cx"]) * 0.5
                cy = (m["right_cy"] + m["left_cy"]) * 0.5

                # 픽셀 단위 좌표 (cx, cy는 ROI 기준임)
                eye_x_px = cx
                eye_y_px = cy

                # 기록 (stim_x, stim_y, eye_x_px, eye_y_px)
                writer.writerow([
                    time.time(),
                    round(stim_x, 3), round(stim_y, 3),
                    round(eye_x_px, 3), round(eye_y_px, 3)
                ])

            cv2.imshow(WIN, canvas)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # ===== 5) 완료 안내 3초 후 종료 =====
        end_t = time.time()
        while time.time() - end_t < 3.0:
            canvas = put_korean_center(canvas, "촬영이 완료되었습니다", 40, (0,255,0))
            cv2.imshow(WIN, canvas)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    print(f"[SAVED] {csv_path}")

if __name__ == "__main__":
    main()
