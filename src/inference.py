import torch
import torchaudio
from src.model import VoiceCleanerModel
from src.audio_utils import load_audio
import numpy as np

class VoiceCleaner:
    def __init__(self):
        self.model_wrapper = VoiceCleanerModel()
        self.target_sr = self.model_wrapper.sample_rate

    def clean_audio(self, audio_path):
        """
        End-to-end cleaning process: Load -> Process -> Return Waveform
        """
        # 1. Load Audio
        print(f"Loading {audio_path}...")
        # Use our robust loader (librosa based) to support mp3 etc better on Windows
        waveform_np, sr = load_audio(audio_path, target_sr=None)
        
        if waveform_np is None:
            raise ValueError(f"Could not load audio file: {audio_path}")
            
        # Convert to Tensor
        waveform = torch.from_numpy(waveform_np)
        
        # Ensure (Channels, Time)
        if waveform.ndim == 1:
            # (Time,) -> (1, Time)
            waveform = waveform.unsqueeze(0)
        
        # 2. Resample if necessary
        if sr != self.target_sr:
            print(f"Resampling from {sr} to {self.target_sr} Hz...")
            resampler = torchaudio.transforms.Resample(sr, self.target_sr)
            waveform = resampler(waveform)
            
        # 3. Ensure proper shape (Batch, Channels, Time)
        # Demucs expects stereo (2 channels). If mono, duplicate.
        if waveform.shape[0] == 1:
            waveform = torch.cat([waveform, waveform], dim=0)
            
        # Add batch dimension
        waveform_in = waveform.unsqueeze(0)
        
        # 4. Inference
        print("Running inference...")
        # Simple chunking logic to avoid OOM on GPU for long files
        # We process in chunks of 10 seconds with little overlap if needed, 
        # but for simplicity, let's try full pass first or use sliding window from torchaudio if we were using appropriate pipeline apply function.
        # Demucs 'separate' directly supports large inputs but might OOM.
        # For this demo, we'll assume reasonable file sizes (< 5 mins).
        
        # Note: The model wrapper's separate method handles normalizing
        separated = self.model_wrapper.separate(waveform_in)
        
        # 5. Extract Vocals
        vocals = self.model_wrapper.get_voice(separated)
        
        # Vocals shape: (1, 2, time)
        vocals = vocals.squeeze(0)
        
        # If original was mono, we can convert back to mono or keep stereo
        # converting to mono for consistency if mostly speech
        vocals_mono = torch.mean(vocals, dim=0, keepdim=True)
        
        return vocals_mono, self.target_sr
