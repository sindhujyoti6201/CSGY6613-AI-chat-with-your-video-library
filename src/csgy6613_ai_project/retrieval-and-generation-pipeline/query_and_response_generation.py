# === Imports ===
import os
import torch
import open_clip
import base64
from PIL import Image
from io import BytesIO
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
import google.generativeai as genai
from collections import defaultdict
import gradio as gr
import tempfile
from moviepy.editor import VideoFileClip
import re

# ---------- 1. Load OpenCLIP ----------
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-B-32', pretrained='laion2b_s34b_b79k'
)
tokenizer = open_clip.get_tokenizer('ViT-B-32')
model = model.to(device).eval()

# ---------- 2. Embed Text Function ----------
def embed_text(text):
    tokens = tokenizer([text]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(tokens)
    return text_features[0].cpu().numpy()

# ---------- 3. Gemini Setup ----------
genai.configure(api_key=os.getenv("GEMINI_API_KEY", ""))
generation_config = {
    "temperature": 0.4,
    "top_k": 20,
    "top_p": 0.9,
    "max_output_tokens": 1024,
}
gemini_model = genai.GenerativeModel(
    model_name="gemini-2.0-flash",
    generation_config=generation_config
)

# ---------- 4. Qdrant Client ----------
qdrant = QdrantClient(host="qdrant", port=6333)

# ---------- 5. Format Context from Qdrant Hits ----------
def format_context(results):
    grouped = defaultdict(list)

    for hit in results:
        payload = hit.payload
        grouped[payload["video_id"]].append({
            "text": payload["text"],
            "start": payload["start"],
            "end": payload["end"],
            "title": payload["title"]
        })

    context_blocks = []
    for video_id, chunks in grouped.items():
        chunks = sorted(chunks, key=lambda x: x["start"])
        merged = []
        current = chunks[0]
        for nxt in chunks[1:]:
            if nxt["start"] <= current["end"] + 120:
                current["end"] = max(current["end"], nxt["end"])
                current["text"] += " " + nxt["text"]
            else:
                merged.append(current)
                current = nxt
        merged.append(current)

        for m in merged:
            context_blocks.append(f"[Video ID: {video_id} | Title: {m['title']} | {m['start']}s - {m['end']}s]: {m['text']}")

    return "\n".join(context_blocks)

# ---------- 6. Ask Gemini with Context ----------
def ask_gemini_with_context(question, context):
    prompt = f"""Answer the question using ONLY the context below.
If relevant information comes from multiple chunks of the SAME video, merge the timestamps (start time should be the start of the first chunk and end time should be the end of the last chunk).
Convert the timestamps to minutes and seconds. Check if the answer is in the context and if not, then give the next best answer.

### Context:
{context}

### Question:
{question}

Return answer as:
Answer: <your answer here>

Video ID: <video_id>
Title: <title>
Start Time: <start> minutes and seconds
End Time: <end> minutes and seconds
"""
    response = gemini_model.generate_content(prompt)
    return response.text.strip()


# ---------- 7. Extract Video Segment ----------
def extract_video_segment(input_path, output_path, start_time, end_time):
    print(f"Extracting video segment from {start_time} to {end_time} seconds.")
    print(f"Input path: {input_path}")
    print(f"Output path: {output_path}")
    clip = VideoFileClip(input_path)
    duration = VideoFileClip(input_path).duration

    if end_time > duration:
        end_time = duration

    clip = clip.subclip(start_time, end_time)
    clip.write_videofile(output_path, codec="libx264", audio_codec="aac")
    clip.close()  # close to release resources

# ---------- 8. Extract Metadata from Answer ----------
def extract_metadata_from_answer(answer):
    video_id_match = re.search(r"Video ID:\s*([^\s]+)", answer)
    start_time_match = re.search(r"Start Time:\s*(\d+)\s*minutes?\s*and\s*(\d+)\s*seconds?", answer)
    end_time_match = re.search(r"End Time:\s*(\d+)\s*minutes?\s*and\s*(\d+)\s*seconds?", answer)

    if not (video_id_match and start_time_match and end_time_match):
        raise ValueError("Could not extract all metadata from the answer.")

    video_id = video_id_match.group(1)
    start_time = int(start_time_match.group(1)) * 60 + int(start_time_match.group(2))
    end_time = int(end_time_match.group(1)) * 60 + int(end_time_match.group(2))

    return video_id, start_time, end_time

# ---------- 9. Process Query ----------
def process_query(question):
    query_vector = embed_text(question)
    results = qdrant.search("video_chunks_multimodal", query_vector, limit=15)
    context_text = format_context(results)
    answer = ask_gemini_with_context(question, context_text)

    video_id, start, end = extract_metadata_from_answer(answer)
    print("======================"*2)
    print(f"Video ID: {video_id}")
    print(f"Start Time: {start}")
    print(f"End Time: {end}")
    print("======================"*2)
    video_path = f"./datasets/videos/{video_id}/{video_id}.mp4"

    print(f"Attempting to clip from {start} to {end} of {video_path}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        trimmed_path = temp_file.name

    extract_video_segment(video_path, trimmed_path, start, end)

    return answer, trimmed_path


# ---------- 10. Gradio Interface ----------
iface = gr.Interface(
    fn=process_query,
    inputs=gr.Textbox(lines=2, placeholder="Enter your question here..."),
    outputs=[
        gr.Textbox(label="Answer"),
        gr.Video(label="Relevant Clip")
    ],
    title="Video RAG",
    description="Ask a question and get an answer grounded in video content, with the relevant clip shown below."
)

if __name__ == "__main__":
    print("Starting Gradio interface...")
    iface.launch(server_name="0.0.0.0", server_port=int(os.getenv("GRADIO_SERVER_PORT", 7860)))