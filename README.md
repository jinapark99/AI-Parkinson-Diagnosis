# AI Parkinson's Disease Diagnosis System

## 📋 Project Overview

This project is an **AI system that supports early diagnosis of Parkinson's disease** through **Eye Tracking** and **Blink Test** analysis.

### 🎯 Key Features
- **Eye Tracking Analysis**: Target tracking for reaction speed and eye coordination measurement
- **Blink Test**: 30-second blink frequency and pattern analysis
- **Real-time Data Collection**: Real-time measurement using OpenCV and MediaPipe
- **Data Visualization**: Intuitive result analysis through Streamlit
- **Trend Analysis**: Visualization of daily measurement result changes

## 🏗️ Project Structure

```
AI-Parkinson-Diagnosis/
├── Data_V2/                          # Data storage folder
│   ├── blinking_test_data/           # Blink test results
│   │   └── YYYY-MM/                  # Year-month folders
│   │       ├── blink_test_[timestamp].csv      # Blink test results
│   │       ├── blink_test_[timestamp].json     # Detailed data (JSON)
│   │       ├── ear_data_[timestamp].csv        # EAR data time series
│   │       └── blink_timestamps_[timestamp].csv # Blink timestamps
│   └── eye_tracking_data/            # Eye tracking test results
│       └── YYYY-MM/                  # Year-month folders
│           ├── eye_tracking_data_[timestamp].csv      # Eye tracking data
│           └── random_eye_tracking_result_[timestamp].json # Analysis results
├── Scripts_V2/                       # Main scripts folder
│   ├── blink_test_V2.py             # Blink test execution script
│   ├── eye_tracking_test_V2.py      # Eye tracking test execution script
│   └── streamlit_eye_analysis_results.py # Result analysis and visualization app
├── legacy/                           # Legacy code and data
└── README.md                         # Project documentation
```

## 🚀 Installation and Execution

### 1. Virtual Environment Setup
```bash
# Create and activate Conda virtual environment
conda create -n mediapipe_env python=3.10
conda activate mediapipe_env

# Install required packages
pip install opencv-python mediapipe numpy pandas streamlit plotly
```

### 2. Run Blink Test
```bash
cd AI-Parkinson-Diagnosis/Scripts_V2
python blink_test_V2.py
```

### 3. Run Eye Tracking Test
```bash
cd AI-Parkinson-Diagnosis/Scripts_V2
python eye_tracking_test_V2.py
```

### 4. Run Result Analysis App
```bash
cd AI-Parkinson-Diagnosis/Scripts_V2
streamlit run streamlit_eye_analysis_results.py
```

## 📊 Data Structure

### Blink Test Data
- **CSV File**: Basic test results (blink count, status, etc.)
- **JSON File**: Detailed data (EAR values, timestamps, metadata)
- **EAR Data**: Left/right eye EAR values for each frame
- **Timestamps**: Blink occurrence timing and intervals

### Eye Tracking Test Data
- **CSV File**: Frame-by-frame eye coordinates and target information
- **JSON File**: Analysis results (reaction time, sync score, movement count)
- **Reaction Time**: Response speed to target changes
- **Sync Score**: Degree of coordination between two eyes

## 🔍 Analysis Indicators

### 1. Latency (Reaction Speed)
- **Normal Range**: 0.3 seconds or less
- **Meaning**: Response speed to visual stimuli
- **Parkinson's Disease Characteristic**: Decreased reaction speed

### 2. Eye Coordination (Binocular Coordination)
- **Normal Range**: 0.9 or higher
- **Meaning**: Degree to which both eyes move together
- **Parkinson's Disease Characteristic**: Decreased coordination ability

### 3. Movement Count (Blink/Tracking)
- **Normal Blink**: 15-20 times/minute
- **Parkinson's Disease**: 5-10 times/minute
- **Meaning**: Muscle control ability

## 📈 Result Visualization

### Streamlit Analysis App
- **Progress Bars**: Display of normalcy for 3 indicators
- **Trend Graph**: Daily measurement result change trends
- **Detailed Information**: Original data and analysis process
- **Encouragement Message**: Personalized advice based on results

### Graph Features
- Y-axis with 20 divisions for precise score display
- Automatic scaling based on first measurement
- Maximum 1-month data display
- Only latest data used for duplicate measurements on same day

## 🎯 Usage Scenarios

### 1. Initial Diagnosis
- Check basic status with blink test
- Measure reaction speed with eye tracking test
- Evaluate overall status with comprehensive score

### 2. Regular Monitoring
- Observe change trends through periodic testing
- Identify improvement/deterioration trends through trend graphs
- Support medical consultation with data

### 3. Research and Education
- Large-scale data collection and analysis
- Early diagnosis research for Parkinson's disease
- Medical staff training material utilization

## 🔧 Technology Stack

- **Computer Vision**: OpenCV, MediaPipe
- **Data Processing**: NumPy, Pandas
- **Web Framework**: Streamlit
- **Visualization**: Plotly
- **Language**: Python 3.10+

## 📝 Precautions

1. **Camera Permission**: Camera access permission required when running tests
2. **Lighting Conditions**: Test recommended under appropriate lighting
3. **Stability**: Minimize movement during testing
4. **Data Backup**: Separate backup recommended for important test results

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📄 License

This project is currently under development and does not have a formal license. Please contact the project maintainers for usage permissions.

## 📞 Contact

If you have questions or suggestions about the project, please create an issue.

---

**⚠️ Not a Medical Diagnostic Tool**: This system is a tool that supports Parkinson's disease diagnosis, and the final diagnosis should follow a specialist's judgment.
