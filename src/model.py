import torch
import torchaudio

class VoiceCleanerModel:
    def __init__(self):
        """
        Initialize the Demucs model from Torchaudio pipelines.
        We use HDEMUCS_HIGH_MUSDB for high quality music source separation.
        Voice/Speech is effectively separated as 'vocals'.
        """
        print("Loading AI Model (Demucs)...")
        try:
            # We use the standard HDEMUCS model trained on MusDB
            self.bundle = torchaudio.pipelines.HDEMUCS_HIGH_MUSDB
            self.model = self.bundle.get_model()
            
            # Use CPU by default for robustness, or CUDA if available
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            self.sample_rate = self.bundle.sample_rate
            
            # Hardcoded labels for MusDB (standard order)
            self.labels = ['drums', 'bass', 'other', 'vocals']
            print(f"Model loaded on {self.device}. Sources: {self.labels}")
            
        except Exception as e:
            print(f"Failed to load Demucs model: {e}")
            raise e

    def separate(self, audio_waveform):
        """
        Separate audio into sources.
        
        Args:
            audio_waveform (torch.Tensor): Audio tensor of shape (batch, channels, time).
            
        Returns:
            torch.Tensor: Separated sources (batch, sources, channels, time).
        """
        # Demucs expects input shape (batch, channels, time)
        # Normalize: Optional but Demucs works best with normalized audio
        ref = audio_waveform.mean(0)
        audio_waveform = (audio_waveform - ref.mean()) / (ref.std() + 1e-8)
        
        with torch.no_grad():
            output = self.model(audio_waveform.to(self.device))
        return output

    def get_voice(self, separated_sources):
        """
        Extract the voice/vocals component.
        """
        # MusDB Source order: drums, bass, other, vocals
        # Vocals is the last source (index 3)
        vocab_idx = 3 
        return separated_sources[:, vocab_idx, :, :]
