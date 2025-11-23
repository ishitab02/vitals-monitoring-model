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

### Installation Steps
1.  **Clone the Repository:**
    ```bash
    git clone www.github.com/ishitab02/vitals-monitoring-model
    cd vitals-monitoring-model
    ```

2.  **Ensure Model Weights Exist:**
    ```bash
    Verify that the `models/UBFC_TSCAN.pth` file is present.
    ```

3.  **Create and Activate Virtual Environment:**
    ```bash
    uv venv
    .\.venv\Scripts\activate
    ```

4.  **Install Dependencies:**
    Install the dependencies from `pyproject.toml`.
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
