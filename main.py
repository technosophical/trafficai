import shutil
import os
import uuid
import librosa
import numpy as np
import pandas as pd

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://technosophical.github.io"],  # Only allow your frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_FOLDER = "uploads"
PROCESSED_FOLDER = "processed"
SAMPLE_FOLDER = "samples"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

def simple_vehicle_detection(audio_path, session_id):
    y, sr = librosa.load(audio_path, sr=16000)
    frame_length = int(1.0 * sr)
    hop_length = frame_length
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    threshold = 0.04
    vehicle_frames = np.where(rms > threshold)[0]
    timestamps = librosa.frames_to_time(vehicle_frames, sr=sr, hop_length=hop_length)

    # Merge detections within 3 seconds
    merged_events = []
    event_start = None

    for t in timestamps:
        if event_start is None:
            event_start = t
        elif t - event_start > 3.0:  # Gap larger than 3 seconds → new event
            merged_events.append(event_start)
            event_start = t
    if event_start is not None:
        merged_events.append(event_start)

    # Now merged_events contains start times of each real vehicle event
    df = pd.DataFrame({'timestamp_sec': merged_events, 'vehicle_detected': ['yes'] * len(merged_events)})
    output_csv = f"{PROCESSED_FOLDER}/{session_id}.csv"
    df.to_csv(output_csv, index=False)
    return df


@app.post("/upload")
async def upload_audio(audio_file: UploadFile = File(...)):
    session_id = str(uuid.uuid4())
    file_location = f"{UPLOAD_FOLDER}/{session_id}.wav"
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(audio_file.file, buffer)
    simple_vehicle_detection(file_location, session_id)
    return {"session_id": session_id, "status": "processing complete"}

@app.get("/process_sample")
async def process_sample():
    sample_file = f"{SAMPLE_FOLDER}/sample.wav"  # Make sure you have a sample file ready
    session_id = f"sample-{uuid.uuid4()}"
    simple_vehicle_detection(sample_file, session_id)
    return {"session_id": session_id, "status": "processing complete (sample)"}

@app.get("/results/{session_id}")
async def get_results(session_id: str):
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
    csv_path = f"{PROCESSED_FOLDER}/{session_id}.csv"
    if not os.path.exists(csv_path):
        return JSONResponse(content={"error": "Session not found."}, status_code=404)
    return FileResponse(csv_path, filename=f"vehicle_events_{session_id}.csv", media_type='text/csv')
