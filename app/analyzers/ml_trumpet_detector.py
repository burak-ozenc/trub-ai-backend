import os
import pickle
import numpy as np
from typing import Optional
from app.analyzers.base_analyzer import BaseAnalyzer
from app.ml.feature_extractor import MLFeatureExtractor
from app.config import settings

class MLTrumpetDetector(BaseAnalyzer):
    """ML-based trumpet detector using trained classifier"""

    def __init__(self):
        super().__init__()
        self.model = None
        self.scaler = None
        self.feature_extractor = None
        self._load_model()

    def _load_model(self):
        """Load trained model from disk"""
        try:
            if not os.path.exists(settings.ML_MODEL_PATH):
                print(f"⚠️  ML model not found at {settings.ML_MODEL_PATH}")
                print("   Run 'python scripts/train_ml_model.py' to train the model")
                return

            with open(settings.ML_MODEL_PATH, 'rb') as f:
                model_data = pickle.load(f)

            self.model = model_data['classifier']
            self.scaler = model_data['scaler']
            self.feature_extractor = model_data['feature_extractor']

            print(f"✅ ML model loaded successfully")

        except Exception as e:
            print(f"❌ Failed to load ML model: {e}")
            self.model = None

    def is_available(self) -> bool:
        """Check if ML model is available"""
        return self.model is not None

    def analyze(self, y: np.ndarray, sr: int) -> dict:
        """
        Analyze audio using trained ML model
        
        Args:
            y: Audio time series
            sr: Sample rate
            
        Returns:
            Dictionary with ML detection results
        """
        if not self.is_available():
            return {
                'ml_available': False,
                'ml_confidence': 0.0,
                'ml_prediction': False,
                'error': 'ML model not available'
            }

        try:
            self.validate_input(y, sr)

            # Extract features
            features = self.feature_extractor.extract_features(y, sr)
            features_scaled = self.scaler.transform(features.reshape(1, -1))

            # Get prediction and probability
            prediction = self.model.predict(features_scaled)[0]
            probabilities = self.model.predict_proba(features_scaled)[0]

            # Confidence is the probability of the predicted class
            confidence = float(probabilities[1])  # Probability of being trumpet (class 1)

            return {
                'ml_available': True,
                'ml_confidence': round(confidence, 3),
                'ml_prediction': bool(prediction),
                'ml_probability_trumpet': round(confidence, 3),
                'ml_probability_non_trumpet': round(1 - confidence, 3)
            }

        except Exception as e:
            return {
                'ml_available': True,
                'ml_confidence': 0.0,
                'ml_prediction': False,
                'error': f'ML prediction failed: {str(e)}'
            }