# 🎺 Trub AI Backend

AI-powered trumpet performance analyzer with real-time audio processing and LLM feedback.

## Features

### Audio Analysis
- **5-Dimension Performance Analysis**: Breath control, tone quality, rhythm/timing, musical expression, note flexibility
- **Trumpet Sound Detection**: Rule-based acoustic analysis with optional ML support
- **LLM Integration**: Natural language feedback via Ollama (local LLM)
- **Audio Enhancement**: Noise reduction, spectral gating, high-pass filtering

### API Features
- RESTful API with automatic documentation
- JWT authentication
- Recording history with audio playback
- Progress tracking and statistics
- File upload support (WAV/MP3)

## Prerequisites

- Python 3.11+
- PostgreSQL 15+
- FFmpeg
- Ollama

## Quick Start

## 1. Clone Repository:
```bash
git clone https://github.com/burak-ozenc/trub-ai-backend.git
cd trub-ai-backend
```

## 2. Create Conda Environment
Create environment from environment.yml:
```bash
conda env create -f environment.yml
conda activate trumpet-analyzer
```


## 3. Install System Dependencies
### Windows:
FFmpeg:

Download from ffmpeg.org/download.html
Extract and add to PATH
Or use: 
```bash
winget install FFmpeg
```

PostgreSQL:

Download from postgresql.org/download/windows
Install using installer (remember your postgres password)

### Ubuntu/Debian
```bash
sudo apt install -y ffmpeg postgresql
```

### macOS
```bash
brew install ffmpeg postgresql
```

## 4. Install Ollama
### Windows:

Download from ollama.ai/download/windows
Run installer
Open terminal and pull model:

```bash
ollama pull deepseek-r1:7b
```

### Linux/macOS:
```bash
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull deepseek-r1:7b
```
Alternative models (lighter/faster):
```bash
ollama pull tinyllama:1.1b
```


## 5. Setup PostgreSQL Database
### Windows (using pgAdmin or psql):
Open pgAdmin or run:
```bash
psql -U postgres
```
Then execute:
```bash
CREATE DATABASE trumpet_analyzer;
CREATE USER trumpet_user WITH PASSWORD 'your_password';
GRANT ALL PRIVILEGES ON DATABASE trumpet_analyzer TO trumpet_user;
```


### Linux/macOS:
```bash
sudo -u postgres psql
```

Then execute:
```bash
CREATE DATABASE trumpet_analyzer;
CREATE USER trumpet_user WITH PASSWORD 'your_password';
GRANT ALL PRIVILEGES ON DATABASE trumpet_analyzer TO trumpet_user;
\q
```

## 6. Configure Environment Variables

Required .env variables:
```bash
DATABASE_URL=postgresql://trumpet_user:your_password@localhost:5432/trumpet_analyzer
SECRET_KEY=your-secret-key
OLLAMA_MODEL=deepseek-r1:7b
UPLOAD_DIR=data/recordings
MAX_FILE_SIZE=50000000
CORS_ORIGINS=http://localhost:3000,http://localhost:5173
```

## 7. Initialize database:
```bash
python scripts/init_db.py
```

## 8.Run application:
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

# API Documentation
Access interactive docs at:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Key Endpoints
### Authentication:

- POST /auth/register - Register user
- POST /auth/login - Login
- GET /auth/me - Current user

### Analysis:

- POST /analysis/comprehensive - Full analysis with LLM
- POST /audio/analyze - Technical analysis only
- POST /audio/analyze-breath - Breath analysis
- POST /audio/analyze-tone - Tone analysis

### Recordings:

- GET /recordings/ - List recordings
- POST /recordings/ - Save recording
- GET /recordings/{id}/audio - Stream audio
- DELETE /recordings/{id} - Delete recording
- GET /recordings/stats/progress - Progress stats

### LLM:

- POST /llm/ask-question - Ask question
- POST /llm/ask-with-context - Ask with audio context



## Configuration
Edit app/config.py for:

- Trumpet frequency range (233Hz - 2118Hz)
- Breath analysis thresholds
- File size limits
- LLM model selection

# License
MIT License - see LICENSE file
Contact
Burak Özenc - GitHub
Project: trub-ai-backend
