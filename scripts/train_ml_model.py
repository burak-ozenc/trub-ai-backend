#!/usr/bin/env python3
"""
Script to train the ML trumpet detection model
"""
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.ml.model_trainer import TrumpetModelTrainer
from app.config import settings

def main():
    """Train the trumpet detection model"""
    print("🎺 Trumpet Detection Model Training")
    print("=" * 40)

    # Initialize trainer
    trainer = TrumpetModelTrainer()

    # Define data directories
    trumpet_dir = os.path.join(settings.ML_TRAINING_DATA_DIR, "trumpet")
    non_trumpet_dir = os.path.join(settings.ML_TRAINING_DATA_DIR, "non_trumpet")

    # Check if directories exist
    if not os.path.exists(trumpet_dir):
        print(f"❌ Trumpet directory not found: {trumpet_dir}")
        print("Please create the directory and add your trumpet audio files.")
        return

    if not os.path.exists(non_trumpet_dir):
        print(f"❌ Non-trumpet directory not found: {non_trumpet_dir}")
        print("Please create the directory and add your non-trumpet audio files.")
        return

    # Count files
    trumpet_files = len([f for f in os.listdir(trumpet_dir)
                         if f.lower().endswith(('.wav', '.mp3', '.m4a', '.flac'))])
    non_trumpet_files = len([f for f in os.listdir(non_trumpet_dir)
                             if f.lower().endswith(('.wav', '.mp3', '.m4a', '.flac'))])

    print(f"📁 Found {trumpet_files} trumpet files")
    print(f"📁 Found {non_trumpet_files} non-trumpet files")

    if trumpet_files == 0 or non_trumpet_files == 0:
        print("❌ Need audio files in both directories to train!")
        return

    try:
        # Prepare dataset
        print("\n🔄 Preparing dataset...")
        X, y = trainer.prepare_dataset(trumpet_dir, non_trumpet_dir)

        # Train model
        print("\n🚀 Training model...")
        results = trainer.train(X, y)

        # Save model
        print("\n💾 Saving model...")
        trainer.save_model(settings.ML_MODEL_PATH)

        print("\n✅ Training completed successfully!")
        print(f"   Final accuracy: {results['accuracy']:.1%}")

        # Usage instructions
        print("\n📋 Next steps:")
        print("1. The trained model is now ready for use")
        print("2. Restart your FastAPI server to load the new model")
        print("3. ML detection will be automatically integrated with existing detection")

    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()