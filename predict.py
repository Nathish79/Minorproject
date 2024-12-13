import pandas as pd
import numpy as np
import glob
import time
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

def analyze_dyslexia_with_timing_and_accuracy():
    """Analyze dyslexia detection with timing and accuracy metrics"""
    start_time = time.time()
    
    print("\n" + "="*50)
    print("DYSLEXIA DETECTION ANALYSIS")
    print("="*50)
    
    # Start processing files
    dyslexic_files = glob.glob('dyslexia_data/dyslexic/*.csv')
    non_dyslexic_files = glob.glob('dyslexia_data/non_dyslexic/*.csv')
    files = dyslexic_files + non_dyslexic_files
    
    if not files:
        print("\nNo CSV files found! Please check if the files exist in:")
        print("- dyslexia_data/dyslexic/")
        print("- dyslexia_data/non_dyslexic/")
        return 0, 0
        
    features_list = []
    labels = []
    
    print("\nProcessing Files:")
    print("-" * 30)
    
    for file in files:
        file_start_time = time.time()
        
        try:
            # Read and process file
            df = pd.read_csv(file)
            
            # Extract features
            features = {
                'mean_lx': df['LX'].mean(),
                'mean_ly': df['LY'].mean(),
                'mean_rx': df['RX'].mean(),
                'mean_ry': df['RY'].mean(),
                'std_lx': df['LX'].std(),
                'std_ly': df['LY'].std(),
                'std_rx': df['RX'].std(),
                'std_ry': df['RY'].std(),
                'fixation_count': len(df[df['LX'].diff() < 0.1]),
                'saccade_length': np.mean(np.sqrt(df['LX'].diff()**2 + df['LY'].diff()**2)),
                'reading_speed': len(df) / df['T'].max(),
                'regression_count': len(df[df['LX'].diff() < 0]),
            }
            
            features_list.append(list(features.values()))
            # Check if file is from dyslexic directory
            is_dyslexic = 1 if file in dyslexic_files else 0
            labels.append(is_dyslexic)
            
            # Calculate processing time for this file
            file_time = time.time() - file_start_time
            
            print(f"\nFile: {file}")
            print(f"Result: {'Dyslexic' if is_dyslexic else 'Non-Dyslexic'}")
            print(f"Processing Time: {file_time:.3f} seconds")
            
        except Exception as e:
            print(f"\nError processing file {file}: {str(e)}")
            continue
    
    if not features_list:
        print("\nNo valid data was processed!")
        return 0, 0
    
    # Convert to numpy arrays
    X = np.array(features_list)
    y = np.array(labels)
    
    # Train model and get accuracy
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    predictions = model.predict(X)
    accuracy = accuracy_score(y, predictions)
    
    # Calculate total time
    total_time = time.time() - start_time
    
    print("\nPERFORMANCE METRICS:")
    print("-" * 30)
    print(f"Total Files Processed: {len(files)}")
    print(f"Average Processing Time per File: {total_time/len(files):.3f} seconds")
    print(f"Total Processing Time: {total_time:.3f} seconds")
    print(f"Model Accuracy: {accuracy*100:.2f}%")
    
    # Get unique classes present in the data
    unique_classes = np.unique(y)
    if len(unique_classes) == 1:
        print("\nNOTE: Only one class present in the dataset")
        class_name = 'Dyslexic' if unique_classes[0] == 1 else 'Non-Dyslexic'
        print(f"All samples are classified as: {class_name}")
    else:
        print("\nDETAILED CLASSIFICATION REPORT:")
        print("-" * 30)
        print(classification_report(y, predictions, target_names=['Non-Dyslexic', 'Dyslexic']))
    
    print("\nDISTRIBUTION:")
    print("-" * 30)
    print(f"Total Participants: {len(y)}")
    print(f"With Dyslexia: {sum(y == 1)} ({sum(y == 1)/len(y)*100:.2f}%)")
    print(f"Without Dyslexia: {sum(y == 0)} ({sum(y == 0)/len(y)*100:.2f}%)")
    
    print("\n" + "="*50)
    
    return total_time, accuracy

if __name__ == "__main__":
    total_time, accuracy = analyze_dyslexia_with_timing_and_accuracy()