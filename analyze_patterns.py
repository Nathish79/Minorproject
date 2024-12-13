import pandas as pd
import numpy as np
import glob
import os
import sys
import matplotlib.pyplot as plt

# Add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from config import *

def load_eye_tracking_data(data_path):
    """Load eye tracking data files from specified path."""
    all_files = glob.glob(data_path)
    data_frames = []
    for file in all_files:
        df = pd.read_csv(file)
        # Add file info to dataframe
        df['file_name'] = os.path.basename(file)
        data_frames.append(df)
    return pd.concat(data_frames, ignore_index=True) if data_frames else pd.DataFrame()

def calculate_velocity(data):
    """Calculate eye movement velocity."""
    dx = data['LX'].diff()
    dy = data['LY'].diff()
    dt = data['T'].diff()
    velocity = np.sqrt(dx**2 + dy**2) / dt
    return velocity

def identify_fixations(data):
    """Identify fixation points based on velocity and duration."""
    velocity = calculate_velocity(data)
    fixations = []
    current_point = None
    duration = 0
    
    for idx, row in data.iterrows():
        # Get time difference
        time_diff = row['T'] - data['T'].iloc[idx-1] if idx > 0 else 0
        
        if velocity.iloc[idx] < SACCADE_VELOCITY_THRESHOLD:
            if current_point is None:
                current_point = row
            duration += time_diff
        else:
            if duration >= FIXATION_THRESHOLD:
                fixations.append({
                    'x': current_point['LX'],
                    'y': current_point['LY'],
                    'duration': duration,
                    'file_name': current_point['file_name']
                })
            current_point = None
            duration = 0
    return fixations

def analyze_reading_patterns(data):
    """Analyze common reading patterns in the eye tracking data."""
    patterns = {
        'linear': 0,          # Left to right reading
        'regression': 0,      # Right to left movements
        'word_skipping': 0,   # Large forward saccades
        'vertical_scanning': 0,# Vertical movements
        'fixation_count': 0,  # Total number of fixations
        'avg_fixation_duration': 0.0  # Average fixation duration
    }
    
    # Calculate detailed metrics
    fixations = identify_fixations(data)
    patterns['fixation_count'] = len(fixations)
    
    if fixations:
        # Calculate average fixation duration
        total_duration = sum(f['duration'] for f in fixations)
        patterns['avg_fixation_duration'] = total_duration / len(fixations)
        
        # Analyze movement patterns
        for i in range(len(fixations) - 1):
            current_fix = fixations[i]
            next_fix = fixations[i + 1]
            
            # Calculate movement direction
            dx = next_fix['x'] - current_fix['x']
            dy = next_fix['y'] - current_fix['y']
            
            # Classify pattern
            if abs(dy) > abs(dx):
                patterns['vertical_scanning'] += 1
            elif dx > 0:
                if dx > 100:  # Threshold for word skipping
                    patterns['word_skipping'] += 1
                else:
                    patterns['linear'] += 1
            else:
                patterns['regression'] += 1
    
    return patterns, fixations

def calculate_additional_metrics(data, fixations):
    """Calculate additional reading metrics."""
    metrics = {
        'reading_speed': 0,
        'saccade_amplitude': 0,
        'regression_ratio': 0,
        'fixation_rate': 0
    }
    
    # Reading speed (words per minute assuming average word length)
    total_time = data['T'].max() - data['T'].min()
    metrics['reading_speed'] = (len(fixations) / total_time) * 60
    
    # Average saccade amplitude
    if len(fixations) > 1:
        saccades = []
        for i in range(len(fixations) - 1):
            dx = fixations[i+1]['x'] - fixations[i]['x']
            dy = fixations[i+1]['y'] - fixations[i]['y']
            saccades.append(np.sqrt(dx**2 + dy**2))
        metrics['saccade_amplitude'] = np.mean(saccades)
    
    return metrics

def print_pattern_analysis(patterns, metrics, group_name):
    """Print detailed analysis results."""
    print(f"\n{group_name} Eye Movement Pattern Analysis:")
    print("=" * (len(group_name) + 28))
    
    # Calculate percentages for movement patterns
    movement_patterns = {k: v for k, v in patterns.items() 
                        if k not in ['fixation_count', 'avg_fixation_duration']}
    total_movements = sum(movement_patterns.values())
    
    if total_movements > 0:
        print("\nReading Patterns:")
        print("-" * 20)
        for pattern, count in movement_patterns.items():
            percentage = (count/total_movements)*100
            print(f"{pattern.replace('_', ' ').title()}: {percentage:.2f}%")
    
    print(f"\nFixation Analysis:")
    print("-" * 20)
    print(f"Total Fixations: {patterns['fixation_count']}")
    print(f"Average Fixation Duration: {patterns['avg_fixation_duration']:.2f} ms")
    
    print(f"\nReading Metrics:")
    print("-" * 20)
    print(f"Reading Speed: {metrics['reading_speed']:.2f} fixations/minute")
    print(f"Average Saccade Amplitude: {metrics['saccade_amplitude']:.2f} units")
    
    if movement_patterns:
        most_common = max(movement_patterns.items(), key=lambda x: x[1])
        print(f"\nMost common pattern: {most_common[0].replace('_', ' ').title()} "
              f"({(most_common[1]/total_movements)*100:.2f}%)")

def visualize_comparison(dyslexic_patterns, non_dyslexic_patterns):
    """Create side-by-side pie charts comparing dyslexic and non-dyslexic patterns."""
    plt.figure(figsize=(15, 7))
    
    # Function to prepare data
    def prepare_data(patterns):
        movement_patterns = {k: v for k, v in patterns.items() 
                           if k not in ['fixation_count', 'avg_fixation_duration']}
        total = sum(movement_patterns.values())
        if total == 0:
            return [], []
        labels = [pattern.replace('_', ' ').title() for pattern in movement_patterns.keys()]
        sizes = [count/total * 100 for count in movement_patterns.values()]
        return labels, sizes
    
    # Prepare data for both groups
    d_labels, d_sizes = prepare_data(dyslexic_patterns)
    nd_labels, nd_sizes = prepare_data(non_dyslexic_patterns)
    
    # Create subplots
    plt.subplot(1, 2, 1)
    if d_sizes:
        plt.pie(d_sizes, labels=d_labels, autopct='%1.1f%%', startangle=90)
        plt.title('Dyslexic Readers')
    
    plt.subplot(1, 2, 2)
    if nd_sizes:
        plt.pie(nd_sizes, labels=nd_labels, autopct='%1.1f%%', startangle=90)
        plt.title('Non-Dyslexic Readers')
    
    plt.suptitle('Eye Movement Pattern Comparison', fontsize=16, y=1.05)
    plt.tight_layout()
    
    # Save the comparison plot
    plt.savefig('eye_movement_comparison.png', bbox_inches='tight')
    plt.close()

def main():
    # Analyze dyslexic readers
    dyslexic_data = load_eye_tracking_data(DYSLEXIA_DATA_PATH)
    dyslexic_patterns = None
    non_dyslexic_patterns = None
    
    if not dyslexic_data.empty:
        dyslexic_patterns, fixations = analyze_reading_patterns(dyslexic_data)
        metrics = calculate_additional_metrics(dyslexic_data, fixations)
        print_pattern_analysis(dyslexic_patterns, metrics, "Dyslexic Readers")
    else:
        print("\nNo dyslexic reader data found!")
    
    # Analyze non-dyslexic readers
    non_dyslexic_data = load_eye_tracking_data(NON_DYSLEXIA_DATA_PATH)
    if not non_dyslexic_data.empty:
        non_dyslexic_patterns, fixations = analyze_reading_patterns(non_dyslexic_data)
        metrics = calculate_additional_metrics(non_dyslexic_data, fixations)
        print_pattern_analysis(non_dyslexic_patterns, metrics, "Non-Dyslexic Readers")
    else:
        print("\nNo non-dyslexic reader data found!")
    
    # Create comparison visualization if both datasets are available
    if dyslexic_patterns and non_dyslexic_patterns:
        visualize_comparison(dyslexic_patterns, non_dyslexic_patterns)
        print("\nComparison visualization has been saved as 'eye_movement_comparison.png'")

if __name__ == "__main__":
    main() 