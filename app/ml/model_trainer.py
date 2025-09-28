import os
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import librosa
from app.ml.feature_extractor import MLFeatureExtractor
from app.config import settings

class TrumpetModelTrainer:
    """Simple trainer for trumpet detection model"""

    def __init__(self):
        self.feature_extractor = MLFeatureExtractor()
        self.scaler = StandardScaler()
        self.classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1  # Use all CPU cores
        )

    def prepare_dataset(self, trumpet_dir: str, non_trumpet_dir: str) -> tuple:
        """
        Load and prepare dataset from audio files
        
        Args:
            trumpet_dir: Directory with trumpet audio files
            non_trumpet_dir: Directory with non-trumpet audio files
            
        Returns:
            X, y arrays for training
        """
        print("Loading trumpet samples...")
        X_trumpet, y_trumpet = self._load_audio_files(trumpet_dir, label=1)

        print("Loading non-trumpet samples...")
        X_non_trumpet, y_non_trumpet = self._load_audio_files(non_trumpet_dir, label=0)

        # Combine datasets
        X = np.vstack([X_trumpet, X_non_trumpet])
        y = np.hstack([y_trumpet, y_non_trumpet])

        print(f"Dataset prepared: {len(X)} samples, {X.shape[1]} features")
        print(f"Trumpet samples: {sum(y)}, Non-trumpet samples: {len(y) - sum(y)}")

        return X, y

    def _load_audio_files(self, directory: str, label: int, max_files: int = None) -> tuple:
        """Load audio files from directory and extract features"""
        X = []
        y = []

        audio_files = [f for f in os.listdir(directory)
                       if f.lower().endswith(('.wav', '.mp3', '.m4a', '.flac'))]

        if max_files:
            audio_files = audio_files[:max_files]

        total_files = len(audio_files)

        for i, filename in enumerate(audio_files):
            if i % 100 == 0:
                print(f"  Processing {i+1}/{total_files} files...")

            try:
                filepath = os.path.join(directory, filename)

                # Load audio file
                y_audio, sr = librosa.load(filepath, sr=22050, duration=10.0)  # Max 10 seconds

                # Skip very short files
                if len(y_audio) < 0.5 * sr:  # Less than 0.5 seconds
                    continue

                # Extract features
                features = self.feature_extractor.extract_features(y_audio, sr)

                X.append(features)
                y.append(label)

            except Exception as e:
                print(f"  Error processing {filename}: {e}")
                continue

        return np.array(X), np.array(y)

    def train(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Train the trumpet detection model"""
        print("\nTraining model...")

        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train classifier
        self.classifier.fit(X_train_scaled, y_train)

        # Evaluate
        y_pred = self.classifier.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"\nModel trained!")
        print(f"Accuracy: {accuracy:.3f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Non-Trumpet', 'Trumpet']))

        # Feature importance (top 10)
        feature_names = self._get_feature_names()
        feature_importance = self.classifier.feature_importances_
        top_features = sorted(zip(feature_names, feature_importance),
                              key=lambda x: x[1], reverse=True)[:10]

        print("\nTop 10 Important Features:")
        for name, importance in top_features:
            print(f"  {name}: {importance:.3f}")

        return {
            'accuracy': accuracy,
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }

    def save_model(self, filepath: str):
        """Save trained model and scaler"""
        model_data = {
            'classifier': self.classifier,
            'scaler': self.scaler,
            'feature_extractor': self.feature_extractor
        }

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"Model saved to: {filepath}")

    def _get_feature_names(self) -> list:
        """Get feature names for interpretability"""
        names = []

        # Spectral features
        names.extend(['spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff'])
        names.extend([f'spectral_contrast_{i}' for i in range(7)])
        names.extend(['zero_crossing_rate', 'rms_energy', 'spectral_flatness'])

        # MFCC features
        names.extend([f'mfcc_{i}' for i in range(13)])

        # Harmonic features
        names.extend(['harmonic_ratio', 'avg_pitch', 'pitch_variation', 'pitch_density', 'chroma_energy'])

        # Temporal features
        names.extend(['onset_rate', 'tempo', 'rms_variation', 'duration'])

        return names