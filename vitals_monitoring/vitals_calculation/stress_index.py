import numpy as np

try:
    from config import (
        STRESS_SDNN_THRESHOLD_HIGH, STRESS_SDNN_THRESHOLD_LOW,
        STRESS_CALCULATION_METHOD, DEBUG_MODE
    )
except ImportError:
    STRESS_SDNN_THRESHOLD_HIGH = 80
    STRESS_SDNN_THRESHOLD_LOW = 30
    STRESS_CALCULATION_METHOD = 'combined'
    DEBUG_MODE = False

# Stress calculation methods

def calculate_stress_index_sdnn(sdnn):
    # Calculate stress index from SDNN (time-domain HRV).
    # High SDNN = Low stress (0), Low SDNN = High stress (100)
    if sdnn is None:
        return 50  # Neutral
    
    if sdnn <= STRESS_SDNN_THRESHOLD_LOW:
        stress = 100.0
    elif sdnn >= STRESS_SDNN_THRESHOLD_HIGH:
        stress = 0.0
    else:
        # Linear interpolation
        stress = 100 - ((sdnn - STRESS_SDNN_THRESHOLD_LOW) / 
                       (STRESS_SDNN_THRESHOLD_HIGH - STRESS_SDNN_THRESHOLD_LOW) * 100)
    
    return float(np.clip(stress, 0, 100))


def calculate_stress_index_lf_hf(lf_hf_ratio):
    # Calculate stress index from LF/HF ratio (frequency-domain HRV).
    # High LF/HF = High stress (100), Low LF/HF = Low stress (0)
    if lf_hf_ratio is None or lf_hf_ratio <= 0:
        return 50  # Neutral
    
    # Logarithmic mapping
    log_lf_hf = np.log(lf_hf_ratio)
    
    # Map log ratio to stress (0-100)
    # Range of log_lf_hf is roughly -0.5 (relaxed) to 1.5 (stressed)
    stress = (log_lf_hf + 0.5) / 2.0 * 100
    
    return float(np.clip(stress, 0, 100))


def calculate_stress_index(sdnn=None, lf_hf_ratio=None, method=None):
    # Calculate comprehensive stress index from HRV metrics.
    if method is None:
        method = STRESS_CALCULATION_METHOD
    
    if method == 'sdnn_only':
        return calculate_stress_index_sdnn(sdnn)
    
    elif method == 'lf_hf_only':
        return calculate_stress_index_lf_hf(lf_hf_ratio)
    
    elif method == 'combined':
        # Average both methods (weighted)
        if sdnn is not None and lf_hf_ratio is not None:
            sdnn_stress = calculate_stress_index_sdnn(sdnn)
            lf_hf_stress = calculate_stress_index_lf_hf(lf_hf_ratio)
            combined = (sdnn_stress * 0.6 + lf_hf_stress * 0.4)
            return float(combined)
        elif sdnn is not None:
            return calculate_stress_index_sdnn(sdnn)
        elif lf_hf_ratio is not None:
            return calculate_stress_index_lf_hf(lf_hf_ratio)
        else:
            return 50.0  # Neutral
    
    else:
        if DEBUG_MODE:
            print(f"Unknown stress method: {method}")
        return 50.0


# STRESS STATUS & INTERPRETATION

def get_stress_status(stress_index):
    # Classify stress level and return status description + color.
    if stress_index is None:
        return "Unknown", (128, 128, 128)
    
    if stress_index < 20:
        return "Very Relaxed 😊", (0, 255, 0)
    elif stress_index < 40:
        return "Relaxed 😌", (100, 255, 0)
    elif stress_index < 60:
        return "Normal 😐", (0, 165, 255)
    elif stress_index < 80:
        return "Stressed 😟", (0, 100, 255)
    else:
        return "Very Stressed 😰", (0, 0, 255)


def get_recommendations(stress_index):
    # Get wellness recommendations based on stress level.
    recommendations = []
    
    if stress_index < 30:
        recommendations.append("✅ Keep up the good work!")
        recommendations.append("💪 Your HRV indicates good relaxation")
    elif stress_index < 60:
        recommendations.append("🧘 Try some deep breathing exercises")
        recommendations.append("💧 Drink some water")
        recommendations.append("🚶 Take a short walk")
    elif stress_index < 80:
        recommendations.append("🛑 You're showing signs of stress")
        recommendations.append("🧘‍♀️ Consider meditation or yoga")
        recommendations.append("🎵 Listen to calming music")
        recommendations.append("😴 Ensure you're getting enough sleep")
    else:
        recommendations.append("🚨 High stress levels detected")
        recommendations.append("🛑 Take a break immediately")
        recommendations.append("🧘‍♂️ Try progressive muscle relaxation")
        recommendations.append("👨‍⚕️ Consider consulting a healthcare provider")
        recommendations.append("🏃 Get some physical exercise")
    
    return recommendations


# STRESS TRENDS & ANALYSIS

def analyze_stress_trend(stress_history, window_size=30):
    # Analyze stress trends over time.
    if not stress_history or len(stress_history) < 2:
        return None
    
    try:
        stress_array = np.array(stress_history[-window_size:])
        
        current = float(stress_array[-1])
        average = float(np.mean(stress_array))
        
        # Calculate trend (linear fit slope)
        x = np.arange(len(stress_array))
        z = np.polyfit(x, stress_array, 1)
        velocity = float(z[0])
        
        if velocity > 1.0:
            trend = 'increasing'
        elif velocity < -1.0:
            trend = 'decreasing'
        else:
            trend = 'stable'
        
        return {
            'current': current,
            'average': average,
            'trend': trend,
            'velocity': velocity,
            'std': float(np.std(stress_array))
        }
    
    except Exception as e:
        if DEBUG_MODE:
            print(f"Trend analysis error: {e}")
        return None


def suggest_intervention(stress_index, stress_trend=None):
    # Suggest interventions based on stress level and trend.
    suggestions = []
    
    # Base interventions
    if stress_index > 70:
        suggestions.append("IMMEDIATE: Practice 5-minute breathing exercise")
        suggestions.append("Reduce caffeine intake")
        suggestions.append("Take a 10-minute break")
    
    elif stress_index > 50:
        suggestions.append("Light yoga or stretching recommended")
        suggestions.append("Check your posture and relax shoulders")
        suggestions.append("Step outside for fresh air")
    
    # Trend-based interventions
    if stress_trend:
        if stress_trend['trend'] == 'increasing' and stress_trend['velocity'] > 5:
            suggestions.insert(0, "⚠️ ALERT: Stress is rapidly increasing!")
            suggestions.append("Consider what triggered this increase")
        
        elif stress_trend['trend'] == 'decreasing':
            suggestions.append("✅ Good: Stress is decreasing")
            suggestions.append("Keep up whatever you're doing!")
    
    return suggestions