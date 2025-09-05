


"""
Fast Flask backend for shelf product detection + grouping.
Optimizations:
 - models loaded once at startup
 - precompute knowledge-base embeddings
 - in-memory crops & batched embeddings
 - FAISS for fast NN search (fallback to sklearn)
 - minimal disk IO during inference
Endpoint:
 POST /predict  form-data: file=image (jpg/png)
Returns JSON with detections and a base64-encoded annotated image.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
app = Flask(__name__)
CORS(app) 

import os
import io
import time
import base64
import glob
import random
from pathlib import Path
from collections import Counter


from PIL import Image
import numpy as np
import cv2
import torch


 # allow all origins

# Try to import FAISS for very fast nearest neighbors; fallback to sklearn
try:
    import faiss
    _HAS_FAISS = True
except Exception:
    _HAS_FAISS = False
    from sklearn.neighbors import NearestNeighbors


# ultralytics YOLO
from ultralytics import YOLO

# Your embedding class (assumed GPU-ready)
from src.img2vec_resnet18 import Img2VecResnet18

# -----------------------
# CONFIG
# -----------------------
MODEL_PATH = os.environ.get("MODEL_PATH", "models/best.pt")  # YOLO weights
KB_PATH = os.environ.get("KB_PATH", "data/knowledge_base/crops/object")  # folder with subfolders per class
N_NEIGHBORS = int(os.environ.get("N_NEIGHBORS", 5))
CONF_THRESH = float(os.environ.get("CONF_THRESH", 0.35))
IOU_THRESH = float(os.environ.get("IOU_THRESH", 0.45))
MAX_SIDE = int(os.environ.get("MAX_SIDE", 1024))  # resize long edge
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PORT = int(os.environ.get("PORT", 5000))

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# -----------------------
# STARTUP: load models & KB
# -----------------------
print(f"[startup] device={DEVICE} loading YOLO model from {MODEL_PATH} ...")
yolo = YOLO(MODEL_PATH)
# Attempt to speed up model (if supported)
try:
    yolo.fuse()
except Exception:
    pass

print("[startup] loading embedding model ...")
img2vec = Img2VecResnet18()  # assume implementation uses given device

# Build knowledge base embeddings (load once)
print("[startup] preparing knowledge-base embeddings ...")
kb_classes = []   # labels per vector
kb_vecs = []      # list of numpy arrays (float32 normalized)

# expected structure: KB_PATH/<class_name>/*.jpg
for class_dir in sorted(Path(KB_PATH).glob("*")):
    if not class_dir.is_dir():
        continue
    class_name = class_dir.name
    for img_path in class_dir.glob("*.jpg"):
        try:
            im = Image.open(str(img_path)).convert("RGB")
            v = img2vec.getVec(im)  # should return numpy array (float32)
            if v is None:
                continue
            # normalize
            v = v.astype('float32')
            v = v / (np.linalg.norm(v) + 1e-8)
            kb_vecs.append(v)
            kb_classes.append(class_name)
            im.close()
        except Exception as e:
            print("KB load error:", img_path, e)

if len(kb_vecs) == 0:
    raise RuntimeError("No knowledge base embeddings found under KB_PATH.")

kb_matrix = np.stack(kb_vecs, axis=0).astype('float32')
print(f"[startup] KB vectors: {kb_matrix.shape}, classes: {len(set(kb_classes))}")

# Build search index: FAISS preferred
if _HAS_FAISS:
    print("[startup] building FAISS index (inner product on normalized vectors)...")
    dim = kb_matrix.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product on L2-normalized vectors ~= cosine similarity
    index.add(kb_matrix)
    def search_nn(query_vecs, k=N_NEIGHBORS):
        # expects query_vecs (nq, dim) float32 & normalized
        D, I = index.search(query_vecs, k)  # D: similarity scores
        return D, I
else:
    print("[startup] FAISS not available. Using sklearn NearestNeighbors fallback (may be slower).")
    nn_model = NearestNeighbors(metric='cosine', n_neighbors=N_NEIGHBORS, algorithm='brute')
    nn_model.fit(kb_matrix)
    def search_nn(query_vecs, k=N_NEIGHBORS):
        # returns distances (cosine) and indices
        dists, idx = nn_model.kneighbors(query_vecs, n_neighbors=k)
        # convert cosine distances to similarity (~1 - dist)
        sims = 1.0 - dists
        return sims, idx

# -----------------------
# Flask app
# -----------------------

def resize_keep_aspect(img_cv2, max_side=MAX_SIDE):
    h, w = img_cv2.shape[:2]
    long = max(h, w)
    if long <= max_side:
        return img_cv2, 1.0
    scale = max_side / long
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(img_cv2, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale

def pil_from_cv2(img_cv2):
    return Image.fromarray(cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB))

def encode_img_to_base64_jpeg(img_cv2, quality=85):
    _, buf = cv2.imencode('.jpg', img_cv2, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    b64 = base64.b64encode(buf).decode('utf-8')
    return b64

def batch_embed_pil(pil_images):
    """Return normalized float32 numpy array shape (N, D)"""
    vecs = []
    batch_size = 16  # adjust based on GPU memory; choose larger for GPU
    for i in range(0, len(pil_images), batch_size):
        batch = pil_images[i:i+batch_size]
        # img2vec should accept PIL and return ndarray (D,) on CPU
        batch_vecs = [img2vec.getVec(im) for im in batch]
        # filter None
        for v in batch_vecs:
            if v is None:
                vecs.append(None)
            else:
                v = v.astype('float32')
                v = v / (np.linalg.norm(v) + 1e-8)
                vecs.append(v)
    # convert to numpy array (keep None as zero vectors and mask later)
    final = []
    mask = []
    for v in vecs:
        if v is None:
            final.append(np.zeros(kb_matrix.shape[1], dtype='float32'))
            mask.append(False)
        else:
            final.append(v)
            mask.append(True)
    return np.stack(final, axis=0).astype('float32'), np.array(mask, dtype=bool)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status":"ok", "device": DEVICE})

@app.route("/predict", methods=["POST"])
def predict():
    start_t = time.time()

    if 'file' not in request.files:
        return jsonify({"error": "no file part"}), 400

    f = request.files['file']
    img_bytes = f.read()
    npimg = np.frombuffer(img_bytes, np.uint8)
    img_cv = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    if img_cv is None:
        return jsonify({"error":"invalid image"}), 400

    # resize for speed
    img_small, scale = resize_keep_aspect(img_cv, MAX_SIDE)

    # Run YOLO detection (in-memory)
    # return results list; we only infer on single image
    with torch.no_grad():
        results = yolo.predict(source=img_small, conf=CONF_THRESH, iou=IOU_THRESH, verbose=False)

    if len(results) == 0:
        return jsonify({"detections": [], "time": time.time()-start_t})

    res = results[0]
    boxes_xyxy = res.boxes.xyxy.cpu().numpy()  # shape (N,4) in resized coords
    scores = res.boxes.conf.cpu().numpy()
    # Convert boxes back to original image coords
    boxes_original = []
    for (x1,y1,x2,y2) in boxes_xyxy:
        x1o = int(x1/scale)
        y1o = int(y1/scale)
        x2o = int(x2/scale)
        y2o = int(y2/scale)
        boxes_original.append([x1o, y1o, x2o, y2o])
    boxes_original = np.array(boxes_original, dtype=int)

    # Crop images in memory (on original-resolution image to preserve detail)
    pil_crops = []
    valid_idx = []
    for i, (x1,y1,x2,y2) in enumerate(boxes_original):
        # clip
        h, w = img_cv.shape[:2]
        x1c = max(0, min(w-1, x1))
        x2c = max(0, min(w, x2))
        y1c = max(0, min(h-1, y1))
        y2c = max(0, min(h, y2))
        if x2c-x1c < 8 or y2c-y1c < 8:
            pil_crops.append(Image.new('RGB', (8,8), (0,0,0)))
            valid_idx.append(False)
            continue
        crop = img_cv[y1c:y2c, x1c:x2c]
        pil = pil_from_cv2(crop)
        pil_crops.append(pil)
        valid_idx.append(True)

    # Batch embed crops
    crop_vecs, mask = batch_embed_pil(pil_crops)  # (N, D), mask boolean

    # Search KB for nearest neighbors
    sims, idxs = search_nn(crop_vecs, k=N_NEIGHBORS)  # sims (N,k), idxs (N,k)
    # Build predictions per crop
    predictions = []
    for i in range(crop_vecs.shape[0]):
        if not mask[i]:
            predictions.append({"label":"unknown", "score":0.0})
            continue
        neigh_idxs = idxs[i]  # indices into kb_classes
        neigh_sims = sims[i]
        # gather class names
        neigh_classes = [kb_classes[j] for j in neigh_idxs]
        # majority vote
        cnt = Counter(neigh_classes)
        # pick most common
        top_label, top_count = cnt.most_common(1)[0]
        # compute mean similarity among neighbors with that label
        sim_mean = float(np.mean([neigh_sims[j] for j,c in enumerate(neigh_classes) if c==top_label]))
        predictions.append({"label": top_label, "score": sim_mean, "raw_neighbors": neigh_classes})

    # Assign colors: same label -> same color
    label_color = {}
    def label_to_color(lbl):
        if lbl not in label_color:
            # deterministic color by seeded RNG on label
            rng = random.Random(hash(lbl) & 0xffffffff)
            label_color[lbl] = [rng.randint(20,235) for _ in range(3)]
        return label_color[lbl]

    # Draw on original image
    vis = img_cv.copy()
    out_dets = []
    for i, box in enumerate(boxes_original):
        x1,y1,x2,y2 = map(int, box.tolist())
        pred = predictions[i]
        lbl = pred["label"]
        score = float(pred["score"])
        color = label_to_color(lbl)
        cv2.rectangle(vis, (x1,y1), (x2,y2), color, 2)
        text = f"{lbl} {score:.2f}"
        # text background
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis, (x1, y1 - th - 6), (x1 + tw + 6, y1), color, -1)
        cv2.putText(vis, text, (x1+3, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        out_dets.append({
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "label": lbl,
            "score": score
        })

    # Encode annotated image to base64
    b64img = encode_img_to_base64_jpeg(vis, quality=85)

    total_time = time.time() - start_t
    resp = {
        "detections": out_dets,
        "annotated_image_base64": b64img,
        "time_seconds": total_time
    }
    return jsonify(resp), 200


if __name__ == "__main__":
    # Recommended: run using gunicorn or uvicorn for production
    print(f"Starting Flask on port {PORT} (device={DEVICE})")
    app.run(host="0.0.0.0", port=PORT, threaded=True)
