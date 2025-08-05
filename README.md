# 🧠 AI-Parkinson-Diagnosis

This project analyzes eye movement data to estimate Parkinson’s disease severity using simple vision-based tracking.

---

## 📁 Project Structure
```
├── scripts/ # Python scripts
│ ├── collect_imu_data.py
│ ├── compare_eye_to_target.py
│ ├── eye_analysis.py
│ ├── eye_tracking_app.py
│ ├── make_dummy_target.py
│ └── merge_eye_and_imu.py
│
├── data/ # Raw and processed CSV data
│ ├── eye_movement_coordinates.csv
│ ├── coordenadas_pelota.csv
│ └── merged_eye_target_MRL.csv
│
├── figures/ # Output visualizations
│ ├── distance_MRL_prediction.png
│ ├── trajectory_x_MRL.png
│ └── trajectory_y_MRL.png
│
├── eyeinfo/ # Eye-tracking config/info
│ └── ...
│
├── .gitignore
└── README.md
```

## 🧪 Features

- ✅ Eye tracking using webcam
- ✅ Target-following mock generator
- ✅ Gaze vs Target comparison (MRL-based error)
- 📊 Visualization with Matplotlib
- 🚧 IMU data integration (in progress)

---

## 📷 Example Visualizations

![Trajectory X](figures/trajectory_x_MRL.png)  
![Trajectory Y](figures/trajectory_y_MRL.png)  
![Distance Error](figures/distance_MRL_prediction.png)

---

## 🚀 How to Run

```bash
# 1. Generate target movement
python3 scripts/make_dummy_target.py

# 2. Run eye-tracking and collect data
python3 scripts/eye_analysis.py

# 3. Compare gaze vs target
python3 scripts/compare_eye_to_target.py
```
