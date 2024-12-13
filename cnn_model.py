import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import glob
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Input
from tensorflow.keras.utils import to_categorical
from config import DATA_PATH_PATTERN
from data_augmentation import augment_data

def prepare_data_for_cnn(file_pattern=DATA_PATH_PATTERN, sequence_length=200, n_features=4):
    """
    Prepare data for CNN by converting time series to 2D representations
    """
    all_files = glob.glob(file_pattern)
    
    print(f"Found {len(all_files)} files")
    if len(all_files) == 0:
        raise ValueError(f"No files found matching pattern: {file_pattern}")
    
    X = []
    y = []
    
    for file in all_files:
        print(f"Processing file: {file}")
        label = 1 if 'P' in file else 0
        
        # Read data
        df = pd.read_csv(file)
        
        # Extract coordinates and normalize
        coords = df[['LX', 'LY', 'RX', 'RY']].values
        
        # Normalize the coordinates
        coords = (coords - np.mean(coords, axis=0)) / np.std(coords, axis=0)
        
        # Pad or truncate sequence to fixed length
        if len(coords) > sequence_length:
            coords = coords[:sequence_length]
        else:
            pad_length = sequence_length - len(coords)
            coords = np.pad(coords, ((0, pad_length), (0, 0)), mode='constant')
        
        additional_features = {
            'reading_pattern': calculate_reading_pattern(df),
            'fixation_duration': calculate_fixation_duration(df),
            'saccade_velocity': calculate_saccade_velocity(df),
            'line_transition': detect_line_transitions(df)
        }
        
        X.append(coords)
        y.append(label)
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"Data shape before reshaping: {X.shape}")
    
    # Reshape for CNN input (samples, height, width, channels)
    X = X.reshape(X.shape[0], sequence_length, n_features, 1)
    
    print(f"Final data shape: {X.shape}")
    print(f"Label distribution: {np.bincount(y)}")
    
    return X, y

def create_cnn_model(input_shape):
    """
    Create CNN model for dyslexia prediction with adjusted architecture
    """
    print(f"Creating model with input shape: {input_shape}")
    
    model = Sequential([
        # Input layer
        Input(shape=input_shape),
        
        # First Convolutional Block
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 1)),  # Only pool along time dimension
        
        # Second Convolutional Block
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 1)),  # Only pool along time dimension
        
        # Third Convolutional Block
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 1)),  # Only pool along time dimension
        
        # Additional Convolution to reduce dimensions
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        
        # Flatten layer
        Flatten(),
        
        # Dense layers
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(2, activation='softmax')
    ])
    
    model.summary()
    return model

def train_cnn_model():
    # Prepare data
    X, y = prepare_data_for_cnn()
    
    if len(X) < 10:
        print("\nApplying data augmentation...")
        try:
            X, y = augment_data(X, y, n_augmented=5)
            print(f"Dataset size after augmentation: {len(X)}")
        except Exception as e:
            print(f"Warning: Data augmentation failed - {str(e)}")
            print("Proceeding with original data...")
    
    y = to_categorical(y)  # Convert labels to one-hot encoding
    
    print(f"Total samples: {len(X)}")
    
    if len(X) < 10:  # If we have very few samples
        print("\nWARNING: Very small dataset detected!")
        print("Options:")
        print("1. Collect more data (recommended)")
        print("2. Use data augmentation")
        print("3. Use cross-validation instead of train-test split")
        print("\nProceeding with simple split without stratification...")
        
        # Simple split without stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.33, random_state=42
        )
    else:
        # Use stratified split for larger datasets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y.argmax(axis=1)
        )
    
    print(f"\nTraining data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    # Create and compile model
    model = create_cnn_model(input_shape=X_train.shape[1:])
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Reduce epochs for small dataset
    epochs = 20 if len(X) < 10 else 50
    
    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=min(32, len(X_train)),  # Adjust batch size for small datasets
        validation_split=0.2,
        verbose=1
    )
    
    # Evaluate model
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"\nTest Accuracy: {test_accuracy*100:.2f}%")
    
    return model, history

def predict_with_cnn(eye_tracking_data, model):
    """
    Make prediction using the trained CNN model
    """
    # Prepare input data
    processed_data = prepare_single_sample(eye_tracking_data)
    
    # Make prediction
    prediction = model.predict(processed_data)
    
    return {
        'has_dyslexia': bool(np.argmax(prediction[0])),
        'confidence': float(prediction[0][1])
    }

def prepare_single_sample(eye_tracking_data, sequence_length=1000):
    """
    Prepare a single sample for prediction
    """
    coords = eye_tracking_data[['LX', 'LY', 'RX', 'RY']].values
    
    # Pad or truncate sequence
    if len(coords) > sequence_length:
        coords = coords[:sequence_length]
    else:
        pad_length = sequence_length - len(coords)
        coords = np.pad(coords, ((0, pad_length), (0, 0)), mode='constant')
    
    # Reshape for CNN input
    coords = coords.reshape(1, sequence_length, 4, 1)
    
    return coords

def visualize_training_history(history):
    """
    Visualize training history
    """
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 4))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def augment_data(X, y):
    # Add noise to existing samples
    X_augmented = []
    y_augmented = []
    for x, label in zip(X, y):
        X_augmented.append(x)
        y_augmented.append(label)
        # Create variations
        for _ in range(5):
            noise = np.random.normal(0, 0.1, x.shape)
            X_augmented.append(x + noise)
            y_augmented.append(label)
    return np.array(X_augmented), np.array(y_augmented)

def ensemble_prediction(models, X):
    predictions = []
    for model in models:
        pred = model.predict(X)
        predictions.append(pred)
    # Take majority vote
    return np.mean(predictions, axis=0) > 0.5

if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Train model
    model, history = train_cnn_model()
    
    # Visualize training history
    visualize_training_history(history) 