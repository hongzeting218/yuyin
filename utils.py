import os
import pandas as pd
import librosa
import numpy as np
import soundfile as sf

def load_data(data_dir):
    """
    Traverse the dataset directory and load file paths and labels.
    """
    file_paths = []
    labels = []
    
    # Walk through the directory
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(('.wav', '.mpeg', '.amr', '.mp3')):
                # Extract label from filename (e.g., "euphoric.wav" -> "euphoric")
                label = os.path.splitext(file)[0].lower()
                # Some filenames might have numbers or other suffixes, but looking at the ls output, 
                # they seem to be exactly the emotion names.
                # Valid labels based on observation: euphoric, joyfully, sad, surprised
                
                # Normalize labels if necessary (e.g., map joyfully to happy if desired, 
                # but for now keep original)
                file_paths.append(os.path.join(root, file))
                labels.append(label)
                
    df = pd.DataFrame({
        'path': file_paths,
        'label': labels
    })
    return df

def noise(data):
    noise_amp = 0.035*np.random.uniform()*np.amax(data)
    data = data + noise_amp*np.random.normal(size=data.shape)
    return data

def stretch(data, rate=0.8):
    return librosa.effects.time_stretch(data, rate=rate)

def shift(data):
    shift_range = int(np.random.uniform(low=-5, high = 5)*1000)
    return np.roll(data, shift_range)

def pitch(data, sampling_rate, pitch_factor=0.7):
    return librosa.effects.pitch_shift(data, sr=sampling_rate, n_steps=pitch_factor)

def extract_features(file_path=None, y_input=None, sr_input=None):
    """
    Extract MFCC, Chroma, Mel, Contrast features from an audio file or raw data.
    If y_input is provided, it uses that instead of loading from file_path.
    """
    try:
        if y_input is not None and sr_input is not None:
            y = y_input
            sr = sr_input
        else:
            # Load audio file
            y, sr = librosa.load(file_path, duration=3, offset=0.5)
        
        # 1. MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfcc_mean = np.mean(mfcc.T, axis=0)
        mfcc_std = np.std(mfcc.T, axis=0)
        
        # 2. Chroma
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma.T, axis=0)
        chroma_std = np.std(chroma.T, axis=0)
        
        # 3. Mel Spectrogram
        mel = librosa.feature.melspectrogram(y=y, sr=sr)
        mel_mean = np.mean(mel.T, axis=0)
        mel_std = np.std(mel.T, axis=0)
        
        # 4. Spectral Contrast
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        contrast_mean = np.mean(contrast.T, axis=0)
        contrast_std = np.std(contrast.T, axis=0)
        
        # Concatenate all features
        features = np.concatenate([
            mfcc_mean, mfcc_std,
            chroma_mean, chroma_std,
            mel_mean, mel_std,
            contrast_mean, contrast_std
        ])
        
        return features
    except Exception as e:
        # print(f"Error processing features: {e}")
        return None

def get_gender_from_pitch(file_path):
    """
    Estimate gender based on fundamental frequency (F0).
    Simple heuristic: 
    - Male: 85-180 Hz
    - Female: 165-255 Hz
    We will use a threshold around 165-175Hz.
    """
    try:
        y, sr = librosa.load(file_path)
        # Extract pitch using pyin (probabilistic YIN)
        f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        
        # Filter out NaN values (unvoiced)
        valid_f0 = f0[~np.isnan(f0)]
        
        if len(valid_f0) == 0:
            return "未知"
            
        mean_f0 = np.mean(valid_f0)
        
        # Threshold (approximate intersection)
        if mean_f0 < 170:
            return "男性"
        else:
            return "女性"
    except Exception as e:
        print(f"Error in gender detection: {e}")
        return "未知"
