# Trub-AI Backend

AI-powered trumpet performance analysis and coaching system backend built with FastAPI.

## Features

- **Audio Analysis**: Real-time trumpet sound detection and performance analysis
- **Breath Control Analysis**: Advanced breathing pattern detection and feedback
- **Tone Quality Assessment**: Harmonic analysis and timbre evaluation
- **AI Coaching**: LLM-powered personalized feedback and recommendations
- **Interactive Q&A**: Chat with AI trumpet teacher for technique guidance
- **Signal Processing**: Advanced audio preprocessing and noise reduction

## Tech Stack

- **Framework**: FastAPI
- **Audio Processing**: LibROSA, NumPy, SciPy
- **AI/LLM**: Ollama (Local LLM)
- **Machine Learning**: Scikit-learn (planned)
- **Validation**: Pydantic
- **Audio Formats**: WAV, MP3, M4A, FLAC

## Installation

### Prerequisites
- Python 3.8+
- FFmpeg
- Ollama

### Setup
```bash
# Clone repository
git clone https://github.com/burak-ozenc/trub-ai-backend
cd trub-ai-backend

# Create virtual environment
python -m venv trumpet-env
source trumpet-env/bin/activate  # Windows: trumpet-env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install Ollama and download model
ollama pull tinyllama:1.1b
```

### Configuration
```bash
# Create data directories
mkdir -p data/recordings
mkdir -p data/ml_training/{trumpet,non_trumpet}

# Set environment variables (optional)
export UPLOAD_DIR=data/recordings
export OLLAMA_MODEL=tinyllama:1.1b
```

## Running the Server

```bash
# Development
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Or directly
python app/main.py
```

Server will be available at `http://localhost:8000`

## API Endpoints

### Core Analysis
- `POST /analysis/comprehensive` - Full audio analysis with AI feedback
- `POST /audio/analyze-breath` - Breath control analysis only
- `POST /audio/analyze-tone` - Tone quality analysis only

### AI Interaction
- `POST /llm/ask-question` - Q&A without audio context
- `POST /llm/ask-with-context` - Q&A with audio analysis context

### Health & Config
- `GET /health` - System health check
- `GET /config` - API configuration info
- `GET /docs` - Interactive API documentation

## Audio Analysis Features

### Trumpet Detection
- Harmonic series verification
- Spectral characteristic analysis
- Pitch stability assessment
- Attack transient detection

### Performance Analysis
- **Breath Control**: Pattern detection, consistency analysis
- **Tone Quality**: Harmonic ratio, spectral clarity
- **Signal Processing**: Bandpass filtering, noise reduction

## Project Structure

```
app/
├── core/           # Models, exceptions, constants
├── api/endpoints/  # API route handlers
├── services/       # Business logic
├── analyzers/      # Audio analysis components
├── utils/          # Utilities and preprocessing
└── ml/            # Machine learning components (future)
```

## License

#TODO
