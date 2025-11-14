import pandas as pd
import numpy as np
import os
import time
from datetime import datetime

try:
    from config import DEBUG_MODE
except ImportError:
    DEBUG_MODE = False

def _debug_print(*args, **kwargs):
    if DEBUG_MODE:
        print("[DataLogger]", *args, **kwargs)

class DataLogger:
    def __init__(self, log_dir="data/logs", log_interval=1.0):
        self.log_dir = log_dir
        self.log_interval = max(0.1, float(log_interval)) # Enforce min interval
        self.last_log_time = 0.0
        self.buffer = []
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_filename = f"vitals_log_{timestamp}.csv"
        self.summary_filename = f"vitals_summary_{timestamp}.txt"
        
        self.log_file_path = os.path.join(self.log_dir, self.log_filename)
        self.summary_file_path = os.path.join(self.log_dir, self.summary_filename)
        
        try:
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
                _debug_print(f"Created log directory: {self.log_dir}")
        except Exception as e:
            print(f"[DataLogger] CRITICAL: Failed to create log directory! {e}")
            
        self.columns = [
            "timestamp", "hr", "hrv_sdnn", "spo2", "stress",
            "signal_quality"
]
        
        self.start_time = time.time()

    def log_measurement(self, **kwargs):
        current_time = time.time()
        
        if (current_time - self.last_log_time) < self.log_interval:
            return
            
        self.last_log_time = current_time
        
        # Create a log entry
        entry = {col: kwargs.get(col, None) for col in self.columns if col != "timestamp"}
        entry["timestamp"] = datetime.now().isoformat()
        
        self.buffer.append(entry)

    def save_and_analyze(self):
        if not self.buffer:
            _debug_print("No data to save.")
            return

        try:
            # Convert buffer to DataFrame
            df = pd.DataFrame(self.buffer, columns=self.columns)
            
            # Save the raw data to CSV
            df.to_csv(self.log_file_path, index=False)
            _debug_print(f"Raw data saved to {self.log_file_path}")

            # Perform analysis
            analysis = self._perform_analysis(df)
            
            # Print analysis to console
            print("\n")
            print(" Vitals Monitoring Session Summary")
            print("=" * 60)
            print(f"Session Duration: {time.time() - self.start_time:.1f} seconds")
            print(f"Total Measurements Logged: {len(df)}")
            print("\nStatistical Summary (of valid data):\n")
            print(analysis)
            print("\n")
            
            # Save analysis summary to text file
            with open(self.summary_file_path, 'w') as f:
                f.write("Vitals Monitoring Session Summary\n")
                f.write("=" * 60 + "\n")
                f.write(f"Session Duration: {time.time() - self.start_time:.1f} seconds\n")
                f.write(f"Total Measurements Logged: {len(df)}\n")
                f.write(f"Log File: {self.log_filename}\n")
                f.write("\nStatistical Summary (of valid data):\n\n")
                f.write(analysis.to_string())
            
            _debug_print(f"Analysis summary saved to {self.summary_file_path}")

        except Exception as e:
            print(f"[DataLogger] ERROR: Failed to save or analyze data: {e}")

    def _perform_analysis(self, df):

        try:
            numeric_df = df.drop(columns=['timestamp']).apply(pd.to_numeric, errors='coerce')
            
            # Drop columns that were entirely None/NaN 
            numeric_df = numeric_df.dropna(axis=1, how='all')
            
            if numeric_df.empty:
                return "No valid numeric data was logged!"

            # Get descriptive statistics
            stats_cols = ['hr', 'hrv_sdnn', 'spo2', 'stress', 'respiratory_rate', 'signal_quality']
            valid_stats_cols = [col for col in stats_cols if col in numeric_df.columns]
            
            if not valid_stats_cols:
                return "No standard vitals data was logged."
                
            summary = numeric_df[valid_stats_cols].describe(percentiles=[.25, .5, .75])
            
            # Format to one decimal place
            return summary.applymap(lambda x: f"{x:.1f}" if isinstance(x, (float, int)) else x)
            
        except Exception as e:
            _debug_print(f"Analysis failed: {e}")
            return f"Error during analysis: {e}"

if __name__ == "__main__":
    # Example usage for testing
    print("Testing DataLogger...")
    logger = DataLogger(log_dir="data/test_logs", log_interval=0.1)
    
    for i in range(10):
        logger.log_measurement(
            hr=np.random.uniform(60, 80),
            hrv_sdnn=np.random.uniform(30, 50),
            stress=np.random.uniform(20, 40),
            signal_quality=np.random.uniform(80, 100)
        )
        time.sleep(0.1)
    
    # Add a bad entry
    logger.log_measurement(hr=None, hrv_sdnn=None)
    
    logger.save_and_analyze()
    print("\nTest complete! Check 'data/test_logs' directory.")