# FastAPI Transcription Starter

Speech-to-text demo using Deepgram's API with Python FastAPI backend and web frontend.

## Prerequisites

- [Deepgram API Key](https://console.deepgram.com/signup?jump=keys) (sign up for free)
- Python 3.9+
- pnpm 10+ (for frontend)

**Note:** This project uses git submodules for the frontend.

## Quick Start

1. **Clone the repository**

```bash
git clone --recurse-submodules https://github.com/deepgram-starters/fastapi-transcription.git
cd fastapi-transcription
```

2. **Install dependencies**

```bash
make init
```

3. **Set your API key**

Create a `.env` file:

```bash
DEEPGRAM_API_KEY=your_api_key_here
```

4. **Run the app**

```bash
make dev
```

### 🌐 Open the App
- Frontend: [http://localhost:8080](http://localhost:8080)
- API Docs: [http://localhost:8080/docs](http://localhost:8080/docs)
- Alternative Docs: [http://localhost:8080/redoc](http://localhost:8080/redoc)

## Features

- Upload audio files or provide URLs for transcription
- Multiple Deepgram model options
- Async/await for better performance
- Automatic OpenAPI/Swagger documentation
- View transcription history

## Architecture

### Backend
- **FastAPI** - Modern async Python web framework
- **Uvicorn** - Lightning-fast ASGI server
- **Async/await** - Non-blocking I/O for better performance
- **Automatic docs** - OpenAPI spec generated automatically

### Frontend
- Pure vanilla JavaScript (no frameworks)
- Deepgram Design System for styling
- IndexedDB for transcription storage

## API Endpoint

### POST /stt/transcribe

**Request (multipart/form-data):**
```
file: <audio_file>  # OR
url: <audio_url>
model: nova-3       # optional
```

**Response:**
```json
{
  "transcript": "...",
  "words": [...],
  "duration": 123.45,
  "metadata": {...}
}
```

## Makefile Commands

```bash
make help      # Show all commands
make init      # Initialize and install
make dev       # Start dev server
make build     # Build frontend
make start     # Start production server
make update    # Update submodules
make clean     # Clean artifacts
```

## Learn More

- [Deepgram STT Documentation](https://developers.deepgram.com/docs/getting-started-with-pre-recorded-audio)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Deepgram Python SDK](https://github.com/deepgram/deepgram-python-sdk)

## License

MIT License
