# 🎺 Trub AI Backend

AI-powered trumpet performance analyzer with real-time audio processing and LLM feedback.

## Features

### Audio Analysis
- **5-Dimension Performance Analysis**: Breath control, tone quality, rhythm/timing, musical expression, note flexibility
- **Trumpet Sound Detection**: Rule-based acoustic analysis with optional ML support
- **LLM Integration**: Natural language feedback via Ollama (local LLM)
- **Audio Enhancement**: Noise reduction, spectral gating, high-pass filtering

### Play-Along Feature (NEW)
- **Song Library**: 30 public domain songs (classical, folk, Christmas)
- **Multiple Difficulties**: Beginner, intermediate, and advanced levels per song
- **Real-Time Practice**: Interactive sheet music with note-by-note validation
- **Session Tracking**: Performance scoring and progress analytics

### API Features
- RESTful API with automatic documentation
- JWT authentication
- Recording history with audio playback
- Progress tracking and statistics
- File upload support (WAV/MP3)
- MIDI and backing track serving

## Prerequisites

- Python 3.11+
- PostgreSQL 15+
- FFmpeg
- Ollama

## Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/burak-ozenc/trub-ai-backend.git
cd trub-ai-backend
```

### 2. Create Conda Environment
```bash
conda env create -f environment.yml
conda activate trumpet-analyzer
```

### 3. Install System Dependencies

**Windows:**
```bash
# FFmpeg
winget install FFmpeg

# PostgreSQL - Download from postgresql.org
```

**Ubuntu/Debian:**
```bash
sudo apt install -y ffmpeg postgresql
```

**macOS:**
```bash
brew install ffmpeg postgresql
```

### 4. Install Ollama

**Windows:**
- Download from ollama.ai/download/windows
- Pull model: `ollama pull deepseek-r1:7b`

**Linux/macOS:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull deepseek-r1:7b
```

### 5. Setup PostgreSQL Database

```sql
CREATE DATABASE trumpet_analyzer;
CREATE USER trumpet_user WITH PASSWORD 'your_password';
GRANT ALL PRIVILEGES ON DATABASE trumpet_analyzer TO trumpet_user;
```

### 6. Configure Environment Variables

Create `.env` file:
```bash
DATABASE_URL=postgresql://trumpet_user:your_password@localhost:5432/trumpet_analyzer
SECRET_KEY=your-secret-key
OLLAMA_MODEL=deepseek-r1:7b
UPLOAD_DIR=data/recordings
MAX_FILE_SIZE=50000000
CORS_ORIGINS=http://localhost:3000,http://localhost:5173
```

### 7. Initialize Database
```bash
python scripts/init_db.py
```

### 8. Run Application
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## API Documentation

Interactive docs at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Key Endpoints

**Authentication:**
- POST /auth/register - Register user
- POST /auth/login - Login
- GET /auth/me - Current user

**Analysis:**
- POST /analysis/comprehensive - Full analysis with LLM
- POST /audio/analyze - Technical analysis only

**Recordings:**
- GET /recordings/ - List recordings
- POST /recordings/ - Save recording
- GET /recordings/{id}/audio - Stream audio
- DELETE /recordings/{id} - Delete recording

**Play-Along (NEW):**
- GET /songs/library - Browse song library
- GET /songs/{id} - Song details
- GET /songs/{id}/midi/{difficulty} - Get MIDI file
- GET /songs/{id}/backing-track - Get backing track audio
- POST /play-along/start - Start practice session
- POST /play-along/submit-performance - Submit session results
- GET /play-along/sessions - Session history

**LLM:**
- POST /llm/ask-question - Ask question
- POST /llm/ask-with-context - Ask with audio context

## Project Structure

```
app/
├── main.py                          # FastAPI app
├── config.py                        # Configuration
├── api/endpoints/
│   ├── auth.py                      # Authentication
│   ├── audio.py                     # Audio upload
│   ├── analysis.py                  # Analysis endpoints
│   ├── songs.py                     # Song library (NEW)
│   └── play_along.py                # Play-along sessions (NEW)
├── services/
│   ├── audio_processor.py           # Audio analysis
│   ├── llm_service.py               # LLM integration
│   └── song_arranger_service.py     # MIDI processing (NEW)
├── analyzers/
│   ├── breath_analyzer.py
│   ├── tone_analyzer.py
│   ├── rhythm_analyzer.py
│   ├── expression_analyzer.py
│   ├── flexibility_analyzer.py
│   └── trumpet_detector.py
├── database/
│   ├── models.py                    # SQLAlchemy models
│   └── crud.py                      # Database operations
└── data/
    ├── recordings/                  # User recordings
    └── songs/                       # Song library (NEW)
        ├── midi/                    # MIDI files
        ├── sheet_music/             # MusicXML files
        └── backing_tracks/          # MP3 backing tracks
```

## Configuration

Edit `app/config.py` for:
- Trumpet frequency range (233Hz - 2118Hz)
- Analysis thresholds
- File size limits
- LLM model selection

## License

MIT License - see LICENSE file

## Contact

Burak Özenc - [GitHub](https://github.com/burak-ozenc)

Project: [trub-ai-backend](https://github.com/burak-ozenc/trub-ai-backend)
