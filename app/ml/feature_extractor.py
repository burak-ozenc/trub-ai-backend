import librosa
import numpy as np
from typing import Dict, Any
import warnings

class MLFeatureExtractor:
    """Simple feature extractor for ML trumpet detection"""

    def __init__(self, sr: int = 22050):
        self.sr = sr

    def extract_features(self, y: np.ndarray, sr: int = None) -> np.ndarray:
        """
        Extract simple but effective features for trumpet detection
        
        Returns:
            Feature vector as numpy array
        """
        if sr is None:
            sr = self.sr

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            features = []

            # Basic spectral features (13 features)
            spectral_features = self._extract_spectral_features(y, sr)
            features.extend(spectral_features)

            # MFCC features (13 features) - great for timbre
            mfcc_features = self._extract_mfcc_features(y, sr)
            features.extend(mfcc_features)

            # Harmonic features (5 features)
            harmonic_features = self._extract_harmonic_features(y, sr)
            features.extend(harmonic_features)

            # Temporal features (4 features)
            temporal_features = self._extract_temporal_features(y, sr)
            features.extend(temporal_features)

            return np.array(features, dtype=np.float32)

    def _extract_spectral_features(self, y: np.ndarray, sr: int) -> list:
        """Extract spectral features"""
        features = []

        # Spectral centroid, bandwidth, rolloff
        centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)[0])
        bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)[0])
        rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)[0])

        features.extend([centroid, bandwidth, rolloff])

        # Spectral contrast (7 bands)
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=6)
        contrast_mean = np.mean(contrast, axis=1)
        features.extend(contrast_mean.tolist())

        # Zero crossing rate
        zcr = np.mean(librosa.feature.zero_crossing_rate(y)[0])
        features.append(zcr)

        # RMS energy
        rms = np.mean(librosa.feature.rms(y=y)[0])
        features.append(rms)

        # Spectral flatness
        flatness = np.mean(librosa.feature.spectral_flatness(y=y)[0])
        features.append(flatness)

        return features

    def _extract_mfcc_features(self, y: np.ndarray, sr: int) -> list:
        """Extract MFCC features (great for instrument identification)"""
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfccs, axis=1)
        return mfcc_mean.tolist()

    def _extract_harmonic_features(self, y: np.ndarray, sr: int) -> list:
        """Extract harmonic-related features"""
        features = []

        # Harmonic-percussive separation
        harmonic, percussive = librosa.effects.hpss(y)

        # Harmonic ratio
        harmonic_energy = np.mean(np.abs(harmonic))
        total_energy = np.mean(np.abs(y))
        harmonic_ratio = harmonic_energy / total_energy if total_energy > 0 else 0
        features.append(harmonic_ratio)

        # Pitch-related features
        try:
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr, threshold=0.1)

            # Extract pitch statistics
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(pitch)

            if pitch_values:
                features.append(np.mean(pitch_values))  # Average pitch
                features.append(np.std(pitch_values))   # Pitch variation
                features.append(len(pitch_values) / pitches.shape[1])  # Pitch density
            else:
                features.extend([0.0, 0.0, 0.0])

        except:
            features.extend([0.0, 0.0, 0.0])

        # Chroma energy
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_energy = np.sum(chroma)
        features.append(chroma_energy)

        return features

    def _extract_temporal_features(self, y: np.ndarray, sr: int) -> list:
        """Extract temporal features"""
        features = []

        # Onset detection
        try:
            onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
            onset_rate = len(onset_frames) / (len(y) / sr)  # Onsets per second
            features.append(onset_rate)
        except:
            features.append(0.0)

        # Tempo estimation
        try:
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            features.append(float(tempo))
        except:
            features.append(0.0)

        # RMS envelope variation
        rms = librosa.feature.rms(y=y)[0]
        rms_variation = np.std(rms) / np.mean(rms) if np.mean(rms) > 0 else 0
        features.append(rms_variation)

        # Duration (normalized to 0-1 for 10 seconds max)
        duration = len(y) / sr
        normalized_duration = min(duration / 10.0, 1.0)
        features.append(normalized_duration)

        return features