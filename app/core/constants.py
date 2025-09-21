# Trumpet acoustic characteristics and detection thresholds

# Trumpet frequency ranges (Hz)
TRUMPET_FUNDAMENTAL_MIN = 165.0  # E3
TRUMPET_FUNDAMENTAL_MAX = 1046.0  # C6
TRUMPET_HARMONIC_MAX = 5000.0    # Upper harmonic limit
TRUMPET_FORMANT_REGIONS = [
    (500, 1200),   # First formant region
    (1200, 2500),  # Second formant region
    (2500, 4000),  # Third formant region
]

# Detection thresholds
TRUMPET_DETECTION_CONFIDENCE_THRESHOLD = 0.6  # Minimum confidence to consider as trumpet
TRUMPET_HARMONIC_RATIO_THRESHOLD = 0.4        # Minimum harmonic content ratio
TRUMPET_SPECTRAL_CENTROID_MIN = 800.0         # Hz - Minimum spectral centroid
TRUMPET_SPECTRAL_CENTROID_MAX = 3000.0        # Hz - Maximum spectral centroid
TRUMPET_ZERO_CROSSING_RATE_MAX = 0.15         # Maximum zero crossing rate for brass
TRUMPET_MIN_DURATION = 0.5                    # Minimum duration in seconds to analyze
TRUMPET_MIN_ENERGY_THRESHOLD = 0.01           # Minimum RMS energy threshold

# New enhanced detection thresholds
TRUMPET_ATTACK_SHARPNESS_MIN = 0.3            # Minimum attack sharpness for brass
TRUMPET_HARMONIC_SERIES_RATIO_MIN = 0.6       # Minimum harmonic series conformity
TRUMPET_PITCH_STABILITY_MIN = 0.7             # Minimum pitch stability ratio
TRUMPET_SPECTRAL_CONSISTENCY_MIN = 0.5        # Minimum spectral consistency
TRUMPET_ONSET_REGULARITY_MIN = 0.4            # Minimum onset regularity

# Spectral features for trumpet identification
TRUMPET_HARMONIC_PEAKS_MIN = 3                # Minimum number of harmonic peaks
TRUMPET_BRIGHTNESS_THRESHOLD = 0.3            # Spectral brightness measure
TRUMPET_ROLLOFF_FREQUENCY_MIN = 2000.0        # Hz - Spectral rolloff point

# Audio quality thresholds
NOISE_FLOOR_THRESHOLD = -40.0                 # dB - Maximum acceptable noise floor
SIGNAL_TO_NOISE_RATIO_MIN = 10.0             # dB - Minimum SNR for good detection