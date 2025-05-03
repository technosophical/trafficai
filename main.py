import shutil
import os
import uuid
import asyncio
import librosa
import numpy as np
import pandas as pd

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

# App setup
app = FastAPI()

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directory setup
UPLOAD_FOLDER = "uploads"
PROCESSED_FOLDER = "processed"
SAMPLE_FOLDER = "samples"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# --- Core Functions ---

def simple_vehicle_detection(audio_path, session_id):
    """Simple RMS energy threshold vehicle detection."""
    y, sr = librosa.load(audio_path, sr=16000)
    frame_length = int(1.0 * sr)
    hop_length = frame_length
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]

    threshold = 0.04  # Adjust sensitivity here
    vehicle_frames = np.where(rms > threshold)[0]
    timestamps = librosa.frames_to_time(vehicle_frames, sr=sr, hop_length=hop_length)

    # Merge detections within 3 seconds
    merged_events = []
    event_start = None
    for t in timestamps:
        if event_start is None:
            event_start = t
        elif t - event_start > 3.0:
            merged_events.append(event_start)
            event_start = t
    if event_start is not None:
        merged_events.append(event_start)

    df = pd.DataFrame({'timestamp_sec': merged_events, 'vehicle_detected': ['yes'] * len(merged_events)})
    output_csv = f"{PROCESSED_FOLDER}/{session_id}.csv"
    df.to_csv(output_csv, index=False)

def compute_bioacoustic_metrics(audio_data, sr):
    """
    Compute bioacoustic metrics from audio data.
    """
    # 1. Bioacoustic Index: Sum of normalized energy in 2–8kHz band
    S = np.abs(librosa.stft(audio_data))
    freqs = librosa.fft_frequencies(sr=sr)
    band_mask = (freqs >= 2000) & (freqs <= 8000)
    bio_index = np.sum(S[band_mask]) / np.sum(S)

    # 2. Average dB level
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    avg_db = np.mean(S_db)

    # 3. dB variability
    db_std = np.std(S_db)

    # 4. Silence Ratio (below -50 dB)
    silence_thresh = -50
    silent_frames = np.sum(S_db < silence_thresh)
    silence_ratio = silent_frames / S_db.size

    return {
        "bioacoustic_index": float(bio_index),
        "average_db": float(avg_db),
        "db_variability": float(db_std),
        "silence_ratio": float(silence_ratio)
    }

# --- API Endpoints ---

@app.post("/upload")
async def upload_audio(audio_file: UploadFile = File(...)):
    """Upload WAV, process, and return session ID when ready."""
    session_id = str(uuid.uuid4())
    file_location = f"{UPLOAD_FOLDER}/{session_id}.wav"

    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(audio_file.file, buffer)

    # Process the uploaded file
    simple_vehicle_detection(file_location, session_id)

    # Wait for the output CSV to be generated
    output_csv = f"{PROCESSED_FOLDER}/{session_id}.csv"
    wait_time = 0
    max_wait = 10  # seconds

    while not os.path.exists(output_csv) and wait_time < max_wait:
        await asyncio.sleep(0.5)
        wait_time += 0.5

    if not os.path.exists(output_csv):
        return JSONResponse(content={"error": "Processing failed or timed out."}, status_code=500)

    return {"session_id": session_id, "status": "processing complete"}

@app.get("/results/{session_id}")
async def get_results(session_id: str):
    """Return vehicle detection results for a given session ID."""
    csv_path = f"{PROCESSED_FOLDER}/{session_id}.csv"
    if not os.path.exists(csv_path):
        return JSONResponse(content={"error": "Session not found."}, status_code=404)

    df = pd.read_csv(csv_path)
    return {
        "vehicles_detected": len(df),
        "timestamps_sec": df['timestamp_sec'].tolist()
    }

@app.get("/download_csv/{session_id}")
async def download_csv(session_id: str):
    """Download the raw CSV results."""
    csv_path = f"{PROCESSED_FOLDER}/{session_id}.csv"
    if not os.path.exists(csv_path):
        return JSONResponse(content={"error": "Session not found."}, status_code=404)

    return FileResponse(csv_path, filename=f"vehicle_events_{session_id}.csv", media_type='text/csv')

# --- Optional: Debug Endpoints ---

@app.get("/list_processed")
async def list_processed_files():
    """(Debug) List all processed session files."""
    return os.listdir(PROCESSED_FOLDER)

from fastapi import HTTPException
import os
import requests
import uuid

@app.get("/process_sample")
async def process_sample():
    session_id = f"sample-{uuid.uuid4()}"
    uploads_dir = "uploads"
    os.makedirs(uploads_dir, exist_ok=True)  # Make sure uploads folder exists

    sample_file_path = os.path.join(uploads_dir, "sample.wav")

    # If the sample doesn't exist locally, download it from GitHub
    if not os.path.exists(sample_file_path):
        try:
            sample_url = "https://raw.githubusercontent.com/technosophical/trafficai/main/samples/sample.wav"
            print(f"Sample file not found locally. Downloading from {sample_url}...")
            response = requests.get(sample_url)
            response.raise_for_status()
            with open(sample_file_path, "wb") as f:
                f.write(response.content)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to download sample file: {str(e)}")

    # Process the local sample file
    simple_vehicle_detection(sample_file_path, session_id)
    return {"session_id": session_id, "status": "processing complete (sample)"}

