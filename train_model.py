import pandas as pd
import numpy as np
import os
import librosa
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
from utils import load_data, extract_features, noise, stretch, shift, pitch

# Configuration
DATA_DIR = r"d:\语言信号处理\keshe2\archive"
MODEL_PATH = "emotion_model.pkl"

def augment_and_extract(file_path):
    """
    Load audio, augment it, and extract features for each augmentation.
    Returns a list of features and the original label count.
    """
    features_list = []
    
    try:
        y, sr = librosa.load(file_path, duration=3, offset=0.5)
        
        # 1. Original
        res1 = extract_features(y_input=y, sr_input=sr)
        if res1 is not None: features_list.append(res1)
        
        # 2. Noise
        data_noise = noise(y)
        res2 = extract_features(y_input=data_noise, sr_input=sr)
        if res2 is not None: features_list.append(res2)
        
        # 3. Stretch
        data_stretch = stretch(y)
        # Note: Time stretch changes duration, but extract_features handles raw input
        res3 = extract_features(y_input=data_stretch, sr_input=sr)
        if res3 is not None: features_list.append(res3)
        
        # 4. Shift
        data_shift = shift(y)
        res4 = extract_features(y_input=data_shift, sr_input=sr)
        if res4 is not None: features_list.append(res4)
        
        # 5. Pitch
        data_pitch = pitch(y, sr)
        res5 = extract_features(y_input=data_pitch, sr_input=sr)
        if res5 is not None: features_list.append(res5)
        
    except Exception as e:
        print(f"Error augmenting {file_path}: {e}")
        
    return features_list

def train():
    print("Loading data...")
    df = load_data(DATA_DIR)
    print(f"Found {len(df)} audio files.")
    
    print("Extracting features with augmentation (this may take a while)...")
    X = []
    y = []
    
    for index, row in df.iterrows():
        # Get augmented features
        feats = augment_and_extract(row['path'])
        for f in feats:
            X.append(f)
            y.append(row['label'])
            
    X = np.array(X)
    y = np.array(y)
    
    print(f"Total training samples after augmentation: {len(X)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Models to test
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "SVM": SVC(kernel='rbf', C=10, gamma='auto', probability=True, random_state=42),
        "MLP (Neural Network)": MLPClassifier(hidden_layer_sizes=(256, 128, 64), max_iter=500, alpha=0.001, solver='adam', random_state=42)
    }
    
    best_model = None
    best_acc = 0.0
    best_name = ""
    
    print("\nTraining and Evaluating Models:")
    for name, model in models.items():
        print(f"\n--- {name} ---")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {acc:.4f}")
        print(classification_report(y_test, y_pred))
        
        if acc > best_acc:
            best_acc = acc
            best_model = model
            best_name = name
            
    print(f"\nBest Model: {best_name} with Accuracy: {best_acc:.4f}")
    
    # Save best model
    print(f"Saving best model to {MODEL_PATH}...")
    joblib.dump(best_model, MODEL_PATH)
    print("Done.")

if __name__ == "__main__":
    train()
