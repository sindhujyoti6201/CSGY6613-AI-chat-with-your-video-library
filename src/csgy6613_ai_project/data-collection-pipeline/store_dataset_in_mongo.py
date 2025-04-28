from datasets import load_dataset
import webvtt
import json
import re
from io import BytesIO
import tempfile
import os
import cv2
import base64
from PIL import Image
from pymongo import MongoClient

def clean_caption_text(text):
    text = text.replace('\n', ' ').strip()
    fillers = {"uh", "um", "you know", "like"} #used a lot as fillers by prof.
    tokens = text.split()
    tokens = [t for t in tokens if t.lower() not in fillers]
    text = " ".join(tokens)

    def remove_repeats(t):
        words = t.split()
        cleaned = []
        seen = set()
        for i in range(len(words) - 2):
            trigram = " ".join(words[i:i+3])
            if trigram in seen:
                continue
            seen.add(trigram)
            cleaned.append(words[i])
        cleaned += words[-2:]
        return " ".join(cleaned)

    text = remove_repeats(text)
    text = re.sub(r'(\b[\w\s]{3,20}\b)( \1)+', r'\1', text)
    return text


def extract_frame_base64(video_path, timestamp):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
    success, frame = cap.read()
    cap.release()
    if not success:
        return None
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def convert_to_seconds(timestamp):
    h, m, s = timestamp.split(":")
    s, ms = s.split(".")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def chunk_by_fixed_window(vtt_str, video_path, window=30):
    with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".vtt") as tmp:
        tmp.write(vtt_str)
        tmp.flush()
        captions = webvtt.read(tmp.name)
    os.remove(tmp.name)

    chunks = []
    current_chunk = ""
    chunk_start = None
    chunk_end = None

    for caption in captions:
        start = convert_to_seconds(caption.start)
        end = convert_to_seconds(caption.end)

        if chunk_start is None:
            chunk_start = start
            chunk_end = end

        if end - chunk_start > window:
            midpoint = (chunk_start + chunk_end) / 2
            frame = extract_frame_base64(video_path, midpoint)
            chunks.append({
                "start": chunk_start,
                "end": chunk_end,
                "text": current_chunk.strip(),
                "frame": frame
            })
            current_chunk = clean_caption_text(caption.text.strip())
            chunk_start = start
            chunk_end = end
        else:
            cleaned_text = clean_caption_text(caption.text.strip())
            current_chunk += " " + cleaned_text
            chunk_end = end

    if current_chunk.strip():
        midpoint = (chunk_start + chunk_end) / 2
        frame = extract_frame_base64(video_path, midpoint)
        chunks.append({
            "start": chunk_start,
            "end": chunk_end,
            "text": current_chunk.strip(),
            "frame": frame
        })

    return chunks


# MongoDB Setup
mongo_uri = os.getenv("MONGO_URI")
client = MongoClient(mongo_uri)  # Connect to MongoDB
db = client["video_and_subtitle_rag_db"]
collection = db["video_and_subtitle_chunks"]
# Ensure the collection is empty before starting
collection.delete_many({}) 

# Dataset Load 
dataset = load_dataset("webdataset", data_files="./datasets/youtube_dataset.tar", streaming=True).with_format("torch")

for sample in dataset["train"]:
    if "en.vtt" not in sample or "info.json" not in sample or "mp4" not in sample:
        continue

    try:
        vtt_raw = sample["en.vtt"]
        vtt_data = vtt_raw.decode("utf-8") if isinstance(vtt_raw, bytes) else vtt_raw

        info_raw = sample["info.json"]
        info = json.loads(info_raw.decode("utf-8")) if isinstance(info_raw, bytes) else (
            json.loads(info_raw) if isinstance(info_raw, str) else info_raw
        )

        video_id = info.get("id", "unknown_id")
        title = info.get("title", "unknown_title")
        filename = f"{video_id}.mp4"

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(sample["mp4"])
            tmp.flush()
            temp_video_path = tmp.name

        chunks = chunk_by_fixed_window(vtt_data, temp_video_path)

        for chunk in chunks:
            chunk.update({
                "video_id": video_id,
                "title": title,
                "filepath": filename
            })
            collection.insert_one(chunk)

        print(f"Inserted {len(chunks)} chunks for {video_id}")
        os.remove(temp_video_path)

    except Exception as e:
        print(f"Failed processing sample: {e}")
        continue
