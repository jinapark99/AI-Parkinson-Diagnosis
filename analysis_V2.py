# analysis_V2.py
import os, glob, argparse, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# ---------- 유틸 ----------
def find_latest_csv(path_or_dir: str):
    if os.path.isfile(path_or_dir):
        return path_or_dir
    files = sorted(glob.glob(os.path.join(path_or_dir, "*.csv")))
    if not files:
        raise FileNotFoundError("지정한 폴더에서 CSV를 찾지 못했습니다.")
    return files[-1]

def moving_avg(x, w=5):
    if w <= 1: return x
    # 끝단 왜곡 줄이기 위해 'reflect' 방식 패딩
    pad = w//2
    xpad = np.pad(x, (pad, pad), mode="reflect")
    ker = np.ones(w) / w
    return np.convolve(xpad, ker, mode="valid")

def estimate_dt(ts):
    dt = np.median(np.diff(ts))
    return float(dt)

def best_lag(stim, eye, max_lag_s, dt):
    """자극과 눈 신호의 상관이 최대가 되는 지연(초)"""
    max_shift = max(1, int(round(max_lag_s / dt)))
    stim = (stim - np.mean(stim))
    eye = (eye - np.mean(eye))
    best_shift, best_corr = 0, -1e9
    for s in range(-max_shift, max_shift+1):
        if s < 0:
            a = stim[:s]
            b = eye[-s:]
        elif s > 0:
            a = stim[s:]
            b = eye[:-s]
        else:
            a = stim
            b = eye
        if len(a) < 3:
            continue
        denom = (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
        corr = float(np.dot(a, b) / denom)
        if corr > best_corr:
            best_corr, best_shift = corr, s
    return best_shift * dt, best_corr

def fit_scale(stim, eye):
    """eye * s 가 stim을 가장 잘 근사하도록 선형 스케일 s 추정 (오프셋 제거 후)"""
    x = stim - np.mean(stim)
    y = eye  - np.mean(eye)
    denom = np.sum(x*x) + 1e-8
    s = float(np.sum(x*y) / denom)
    return s

def metrics(y_true, y_pred):
    err = y_pred - y_true
    mae  = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    return mae, rmse

# ---------- 분석 메인 ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data", help="CSV 경로나 폴더(기본: data 폴더 최신 파일)")
    ap.add_argument("--smooth", type=int, default=5, help="이동평균 창(샘플)")
    ap.add_argument("--maxlag", type=float, default=1.0, help="최대 지연 탐색 범위(초)")
    ap.add_argument("--save", action="store_true", help="리포트/그림 저장")
    args = ap.parse_args()

    csv_path = find_latest_csv(args.csv)
    df = pd.read_csv(csv_path)

    # 컬럼 체크
    need_cols = [
        "timestamp","stim_x","stim_y","inter_ocular_px",
        "offset_x_monitor_right","offset_y_monitor_right",
        "offset_x_monitor_left","offset_y_monitor_left"
    ]
    for c in need_cols:
        if c not in df.columns:
            raise ValueError(f"CSV에 '{c}' 컬럼이 없습니다.")

    # numpy로 변환
    ts   = df["timestamp"].to_numpy(dtype=float)
    sx   = df["stim_x"].to_numpy(dtype=float)
    sy   = df["stim_y"].to_numpy(dtype=float)
    rx   = df["offset_x_monitor_right"].to_numpy(dtype=float)
    ry   = df["offset_y_monitor_right"].to_numpy(dtype=float)
    lx   = df["offset_x_monitor_left"].to_numpy(dtype=float)
    ly   = df["offset_y_monitor_left"].to_numpy(dtype=float)

    # 시간 간격 추정
    dt = estimate_dt(ts)  # 초
    fps_est = 1.0 / dt if dt > 0 else np.nan

    # 중심 기준 변위로 정렬(자극은 중앙 기준, 눈은 face_center 기준 오프셋이므로 이미 중앙 기준)
    # 자극 중앙(세션 중간값 사용)
    cx, cy = np.median(sx), np.median(sy)
    sdx = sx - cx
    sdy = sy - cy

    # 좌/우 눈을 평균한 '양안' 신호도 준비
    bx = 0.5 * (rx + lx)
    by = 0.5 * (ry + ly)

    # 스무딩(옵션)
    if args.smooth > 1:
        sdx = moving_avg(sdx, args.smooth)
        sdy = moving_avg(sdy, args.smooth)
        rx  = moving_avg(rx,  args.smooth); ry = moving_avg(ry, args.smooth)
        lx  = moving_avg(lx,  args.smooth); ly = moving_avg(ly, args.smooth)
        bx  = moving_avg(bx,  args.smooth); by = moving_avg(by, args.smooth)
        # ts 길이를 맞추기 위해 앞/뒤 잘림이 있는 경우를 대비
        n = min(len(ts), len(sdx))
        ts, sdx, sdy, rx, ry, lx, ly, bx, by = ts[:n], sdx[:n], sdy[:n], rx[:n], ry[:n], lx[:n], ly[:n], bx[:n], by[:n]

    # ----- 축별로 지연/스케일 보정 추정 (양안 기준) -----
    lag_x, corr_x = best_lag(sdx, bx, args.maxlag, dt)
    lag_y, corr_y = best_lag(sdy, by, args.maxlag, dt)

    # 지연 보정 적용 함수
    def apply_lag(a, lag_s, dt):
        shift = int(round(lag_s/dt))
        if shift > 0:
            return np.concatenate([a[shift:], np.full(shift, np.nan)])
        elif shift < 0:
            shift = -shift
            return np.concatenate([np.full(shift, np.nan), a[:-shift]])
        else:
            return a.copy()

    bx_lag = apply_lag(bx, lag_x, dt)
    by_lag = apply_lag(by, lag_y, dt)

    # NaN 제거를 위해 공통 유효 구간 마스크
    mask_x = ~np.isnan(bx_lag)
    mask_y = ~np.isnan(by_lag)

    # 스케일(진폭) 보정
    scale_x = fit_scale(sdx[mask_x], bx_lag[mask_x])
    scale_y = fit_scale(sdy[mask_y], by_lag[mask_y])

    bx_corr = bx_lag * scale_x
    by_corr = by_lag * scale_y

    # 성능 지표
    mae_x, rmse_x = metrics(sdx[mask_x], bx_corr[mask_x])
    mae_y, rmse_y = metrics(sdy[mask_y], by_corr[mask_y])

    # ----- 좌/우 눈도 동일 절차로 별도 계산 (참고용) -----
    rx_lag = apply_lag(rx, lag_x, dt); lx_lag = apply_lag(lx, lag_x, dt)
    ry_lag = apply_lag(ry, lag_y, dt); ly_lag = apply_lag(ly, lag_y, dt)

    rx_corr = rx_lag * scale_x; lx_corr = lx_lag * scale_x
    ry_corr = ry_lag * scale_y; ly_corr = ly_lag * scale_y

    # ===== 그래프 =====
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(ts, sdx, label="Stim X", linewidth=1.5)
    axes[0].plot(ts, bx_corr, label=f"Eye X (both, lag={lag_x:.3f}s, scale={scale_x:.2f})", linewidth=1.0)
    axes[0].set_ylabel("X (px)")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(ts, sdy, label="Stim Y", linewidth=1.5)
    axes[1].plot(ts, by_corr, label=f"Eye Y (both, lag={lag_y:.3f}s, scale={scale_y:.2f})", linewidth=1.0)
    axes[1].set_ylabel("Y (px)")
    axes[1].set_xlabel("Timestamp (s)")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.25)

    fig.suptitle("Stimulus vs Eye (lag & scale corrected)")
    fig.tight_layout(rect=[0,0,1,0.95])

    # 오차 그래프
    fig2, axes2 = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    axes2[0].plot(ts[mask_x], bx_corr[mask_x] - sdx[mask_x], label="Error X (Eye - Stim)")
    axes2[1].plot(ts[mask_y], by_corr[mask_y] - sdy[mask_y], label="Error Y (Eye - Stim)")
    axes2[0].axhline(0, color='k', linewidth=0.8); axes2[1].axhline(0, color='k', linewidth=0.8)
    axes2[0].set_ylabel("px"); axes2[1].set_ylabel("px"); axes2[1].set_xlabel("Timestamp (s)")
    axes2[0].legend(); axes2[1].legend()
    axes2[0].grid(True, alpha=0.25); axes2[1].grid(True, alpha=0.25)
    fig2.suptitle("Tracking Error")
    fig2.tight_layout(rect=[0,0,1,0.95])

    # ===== 리포트 =====
    basename = os.path.splitext(os.path.basename(csv_path))[0]
    report_dir = os.path.join("reports", basename)
    os.makedirs(report_dir, exist_ok=True)
    summary_txt = os.path.join(report_dir, "summary.txt")

    summary = f"""
File: {csv_path}
Samples: {len(ts)}, dt~{dt:.4f}s (fps~{(1/dt if dt>0 else float('nan')):.2f})

Lag & Scale (both eyes):
  X: lag={lag_x:.3f}s, scale={scale_x:.3f}, corr≈{corr_x:.3f},  MAE={mae_x:.3f}px, RMSE={rmse_x:.3f}px
  Y: lag={lag_y:.3f}s, scale={scale_y:.3f}, corr≈{corr_y:.3f},  MAE={mae_y:.3f}px, RMSE={rmse_y:.3f}px
"""

    print(summary)
    if args.save:
        fig.savefig(os.path.join(report_dir, "stim_vs_eye.png"), dpi=150)
        fig2.savefig(os.path.join(report_dir, "error.png"), dpi=150)
        with open(summary_txt, "w", encoding="utf-8") as f:
            f.write(summary.strip() + "\n")

    plt.show()

if __name__ == "__main__":
    main()
