import librosa
import numpy as np
import warnings
from scipy.signal import butter, filtfilt
import noisereduce as nr
from app.config import settings
from app.core.exceptions import AudioProcessingError

class AudioPreprocessor:
    """Handles audio preprocessing operations"""

    def __init__(self):
        self.low_cutoff = settings.TRUMPET_LOW_FREQ
        self.high_cutoff = settings.TRUMPET_HIGH_FREQ

    def load_and_preprocess(self, file_path: str) -> tuple[np.ndarray, int]:
        """
        Load audio file and apply preprocessing pipeline
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Tuple of (processed_audio, sample_rate)
        """
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                # Load audio
                y, sr = librosa.load(file_path, sr=settings.AUDIO_SAMPLE_RATE)

                # Apply preprocessing pipeline
                y_processed = self._preprocess_pipeline(y, sr)

                return y_processed, sr

        except Exception as e:
            raise AudioProcessingError(f"Failed to load and preprocess audio: {str(e)}")

    def _preprocess_pipeline(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Apply complete preprocessing pipeline"""
        # Step 1: Bandpass filter for trumpet frequency range
        y_filtered = self.apply_bandpass_filter(y, sr)

        # Step 2: Remove background noise
        y_denoised = self.remove_background_noise(y_filtered, sr)

        # Step 3: Enhance trumpet signal
        y_enhanced = self.enhance_trumpet_signal(y_denoised)

        return y_enhanced

    def apply_bandpass_filter(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Apply bandpass filter for trumpet frequency range"""
        try:
            nyquist = 0.5 * sr
            low = self.low_cutoff / nyquist
            high = self.high_cutoff / nyquist

            # Ensure filter parameters are valid
            if low >= 1.0 or high >= 1.0 or low <= 0 or high <= 0:
                # Fallback to no filtering if parameters are invalid
                return y

            b, a = butter(N=4, Wn=[low, high], btype='band')
            return filtfilt(b, a, y)

        except Exception as e:
            print(f"Bandpass filter warning: {e}")
            return y  # Return original if filtering fails

    def remove_background_noise(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Remove background noise using spectral subtraction"""
        try:
            # Use first 0.5 seconds as noise profile
            noise_duration = min(0.5, len(y) / sr * 0.1)  # Max 10% of audio
            noise_sample = y[:int(noise_duration * sr)]

            if len(noise_sample) > 0:
                return nr.reduce_noise(y=y, sr=sr, y_noise=noise_sample)
            else:
                return y

        except Exception as e:
            print(f"Noise reduction warning: {e}")
            return y  # Return original if noise reduction fails

    def enhance_trumpet_signal(self, y: np.ndarray) -> np.ndarray:
        """Enhance trumpet signal using harmonic/percussive separation"""
        try:
            # Use percussive component which often contains the attack characteristics
            return librosa.effects.percussive(y, margin=1.0)
        except Exception as e:
            print(f"Signal enhancement warning: {e}")
            return y  # Return original if enhancement fails