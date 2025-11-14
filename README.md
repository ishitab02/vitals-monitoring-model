# Vitals Monitoring System: Real-Time rPPG Analysis

This project implements a real-time, camera-based system for monitoring vital signs, leveraging deep learning (TSCAN model) and traditional signal processing techniques (CHROM, HRV metrics). It extracts the Blood Volume Pulse (BVP) signal from facial video to derive physiological metrics.

## Features
The system captures, processes, and displays the following key metrics:
- Heart Rate (HR): Beats Per Minute (BPM).
- Heart Rate Variability (HRV): Time-domain metrics (SDNN, RMSSD).
- Stress Index: Calculated from HRV (SDNN) and displayed as a percentage (0-100%).
- SpO2 (Simulated): Blood Oxygen Saturation (Calibrated for demo purposes).
- Signal Quality: An objective score (0-100%) indicating the reliability of the BVP signal extraction.
- Visual Feedback: Real-time BVP waveform and dynamic ROI (Region of Interest) tracking.

## Project Structure

The project follows a standard Python packaging structure.

```
VITALS-MONITORING-MODEL/
├── data/
│   └── logs/                 # Stores CSV logs of vitals
├── models/
│   └── UBFC_TSCAN.pth        # Pre-trained TSCAN Model Weights
├── vitals_monitoring/        # The main Python package
│   ├── __init__.py
│   ├── data_logger.py        # Handles logging clean metrics to CSV
│   ├── fusion.py             # Tools for signal standardization
│   ├── main.py               # Main application loop and CV interface
│   └── overlay.py            # Functions for drawing UI, text, and waveforms
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── chrom.py          # CHROM algorithm logic (samples, better skin mask)
│   │   └── roi_extraction.py # MediaPipe integration and face detection (ROI)
│   └── vitals_calculation/
│       ├── __init__.py
│       ├── heart_rate.py
|       |── rppg.py           # real time rPPG analysis
│       ├── hrv_metrics.py    # SDNN, RMSSD calculation
│       ├── spo2.py           # SpO2 estimation logic 
│       ├── stress_index.py   # Stress index calculation
|       └── tscan_model.py    # TSCAN model architecture and implementation
├── .gitignore
├── config.py                 # Central configuration file 
└── pyproject.toml
```
## Setup and Installation
This project uses uv for fast, reliable dependency management.

### Prerequisites
- Python 3.10+
- uv (Installed globally: pip install uv)
- A Webcam

### Installation Steps
1.  **Clone the Repository:**
    ```bash
    git clone www.github.com/ishitab02/vitals-monitoring-model
    cd vitals-monitoring-model
    ```

2.  **Ensure Model Weights Exist:**
    ```bash
    Verify that the `models/UBFC_TSCAN.pth` file is present. (If you need a new checkpoint, place it here.)
    ```

3.  **Create and Activate Virtual Environment:**
    ```bash
    uv venv
    .\.venv\Scripts\activate  # On Windows PowerShell
    # or source .venv/bin/activate on Linux/macOS
    ```

4.  **Install Dependencies:**
    Install the project dependencies and the development tools defined in `pyproject.toml`.
    ```bash
    uv sync --active
    ```

## Usage and Controls

Run the application using the installed Python module interface.

```bash
python -m vitals_monitoring.main
```

## Output Metrics
The main display shows the following metrics:

- HR: Heart Rate (bpm).
- SDNN/RMSSD: Heart Rate Variability metrics (ms).
- Stress: Calculated stress level (0-100%).
- SpO2: Blood Oxygen Saturation (Demo calibrated).

## Output and Analysis
Upon pressing `Q` or when the program is interrupted, the system automatically saves the following files to the `data/logs` directory:

- `vitals_log_*.csv`: A raw log file containing time-series data for all calculated metrics (hr, sdnn, spo2, stress, signal_quality)
- `vitals_summary_*.txt`: A statistical analysis summary showing the mean, min, max, and standard deviation for the entire session
