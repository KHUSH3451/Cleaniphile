import librosa
import soundfile as sf
import torch
import numpy as np
import matplotlib.pyplot as plt
import io

def load_audio(file_path, target_sr=None):
    """
    Load an audio file.
    
    Args:
        file_path (str): Path to the audio file.
        target_sr (int): Target sampling rate. If None, uses the native sampling rate.
        
    Returns:
        tuple: (audio_waveform, sampling_rate)
               audio_waveform is a numpy array (channels, time) or (time,)
    """
    try:
        y, sr = librosa.load(file_path, sr=target_sr, mono=False)
        return y, sr
    except Exception as e:
        print(f"Error loading audio: {e}")
        return None, None

def save_audio(file_path, audio_data, sr):
    """
    Save audio to a file.
    
    Args:
        file_path (str): Path where the file will be saved.
        audio_data (numpy array): Audio waveform (channels, time) or (time,).
        sr (int): Sampling rate.
    """
    try:
        # soundfile expects (time, channels)
        if audio_data.ndim > 1:
            if audio_data.shape[0] < audio_data.shape[1]: 
                # Assuming input is (channels, time), transpose to (time, channels)
                audio_data = audio_data.T
                
        sf.write(file_path, audio_data, sr)
        return True
    except Exception as e:
        print(f"Error saving audio: {e}")
        return False

def plot_waveform(audio_data, sr, title="Waveform"):
    """
    Generate a matplotlib figure of the waveform.
    """
    fig, ax = plt.subplots(figsize=(10, 3))
    
    # If stereo, just plot one channel or average for visualization
    if audio_data.ndim > 1:
        y_to_plot = np.mean(audio_data, axis=0)
    else:
        y_to_plot = audio_data
        
    times = np.linspace(0, len(y_to_plot) / sr, num=len(y_to_plot))
    ax.plot(times, y_to_plot)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    plt.tight_layout()
    return fig

def plot_spectrogram(audio_data, sr, title="Spectrogram"):
    """
    Generate a spectrogram figure.
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    
    if audio_data.ndim > 1:
        y_to_plot = np.mean(audio_data, axis=0)
    else:
        y_to_plot = audio_data
        
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y_to_plot)), ref=np.max)
    img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', ax=ax)
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title(title)
    return fig
