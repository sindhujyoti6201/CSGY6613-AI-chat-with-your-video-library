# === Imports ===
import torch
import open_clip
import base64
from PIL import Image
from io import BytesIO
import os
from pymongo import MongoClient
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

# ---------- 0. MongoDB Setup ----------
mongo_uri = os.getenv("MONGO_URI")
client = MongoClient(mongo_uri)
db = client["video_and_subtitle_rag_db"]
collection = db["video_and_subtitle_chunks"]

# ---------- 1. Load OpenCLIP ----------
device = "cuda" if torch.cuda.is_available() else "cpu"

model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-B-32', pretrained='laion2b_s34b_b79k'
)
tokenizer = open_clip.get_tokenizer('ViT-B-32')
model = model.to(device).eval()

# ---------- 2. Embed Functions ----------
def embed_text(text):
    tokens = tokenizer([text]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(tokens)
    return text_features[0].cpu().numpy()

def embed_image_from_base64(b64):
    img = Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")
    img_tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(img_tensor)
    return image_features[0].cpu().numpy()

# ---------- 3. Load from MongoDB & Embed ----------
points = []
cursor = collection.find({})
for i, doc in enumerate(cursor):
    try:
        if not doc.get("frame") or not doc.get("text"):
            continue

        text_vec = embed_text(doc["text"])
        img_vec = embed_image_from_base64(doc["frame"])
        # Check if we can give more focus on the text part
        # text_vec = text_vec * 0.7 + img_vec * 0.3
        multimodal_vec = (text_vec + img_vec) / 2

        points.append(PointStruct(
            id=i,
            vector=multimodal_vec.tolist(),
            payload={
                "text": doc["text"],
                "video_id": doc.get("video_id"),
                "title": doc.get("title"),
                "start": doc.get("start"),
                "end": doc.get("end"),
                "filepath": doc.get("filepath")
            }
        ))
    except Exception as e:
        print(f"Error in doc {i}: {e}")
        continue

# ---------- 4. Upload to Qdrant ----------
if points:
    # Qdrant Setup
    qdrant = QdrantClient(host="qdrant", port=6333)

    qdrant.recreate_collection(
        collection_name="video_chunks_multimodal",
        vectors_config=VectorParams(size=len(points[0].vector), distance=Distance.COSINE)
    )

    qdrant.upsert(collection_name="video_chunks_multimodal", points=points)

    print(f"Uploaded {len(points)} multimodal vectors to Qdrant.")
else:
    print("No valid data points found for embedding and upload.")
