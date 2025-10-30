VISOR: An AI-Powered Guiding Shield for Vision (SLM-first)

Overview
--------
VISOR is a lightweight, multimodal MVP for assistive guidance:
- Object detection with YOLOv8 (configurable; defaults to yolov8n)
- Image captioning with BLIP base (configurable)
- VQA with BLIP VQA base (improved defaults)
- Scene narrative via small language model (configurable; default: google/flan-t5-small)
- Text-to-speech in the browser (speaks narrative when available)
- Minimal FastAPI backend + HTML/JS frontend

Flow
----
1) Frontend opens webcam and periodically captures frames.
2) Frames are POSTed to /analyze.
3) Backend runs YOLO + BLIP caption (+ BLIP VQA default question), fuses outputs
   with a small language model to produce a concise narrative, and returns:
   - narrative, caption, vqa_answer, detections [{class_name, confidence, xyxy}], device
4) Frontend overlays boxes, shows text, and speaks the caption.

Datasets (for training/eval in the notebook)
--------------------------------------------
- COCO Captions 2017: captions_train/val2017.json, images train/val
- VQA v2.0: val2014 Q/A JSONs (uses COCO images)
- Flickr30k Entities: manual download if grounding needed later

Evaluation (backend.ipynb)
--------------------------
- YOLO: mAP on coco128 (quick sanity). Full COCO val requires YOLO-format labels.
- Captions: BLEU-1/2/3/4, METEOR, ROUGE_L, CIDEr via pycocoevalcap
- VQA: small exact-match demo (note val2014 vs val2017 filename differences)

Requirements
------------
- Python 3.11+ recommended
- Node not required (static frontend served by FastAPI)

Setup (fresh venv recommended)
------------------------------
cd /Users/shubh/Desktop/VISOR
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

Windows setup notes
-------------------
1) Create venv (PowerShell):
   python -m venv .venv
   .venv\\Scripts\\Activate.ps1
2) Install PyTorch for your system (CPU example):
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
3) Install the rest:
   pip install -r requirements.txt

Run the Backend + Frontend
--------------------------
source .venv/bin/activate
uvicorn backend:app --reload
Open http://127.0.0.1:8000/ui
Click "Start Camera" and allow permissions.
About page: http://127.0.0.1:8000/ui/about.html

Notebook (training/eval & downloads)
------------------------------------
- Open backend.ipynb
- Run the "MUST RUN FIRST" cell to disable TF/Flax
- Run dataset setup cells to download COCO/VQA
- Run evaluation cells (YOLO demo mAP, caption metrics, optional VQA)

Notes
-----
- By default VQA uses the prompt "What is in front of me?" if none is supplied.
- Beam search for captioning can be toggled in the notebook; the backend uses greedy decoding for latency.
- The backend now returns a `narrative` synthesized from caption + detections using a small language model for clearer speech output.
- VQA generation length increased modestly for better answers (max_new_tokens=20).
- Assistant vs Guide: "Assistant mode" speaks periodic summaries; "Guide me" gives short obstacle cues. When both are on, Assistant pauses while Guide speaks.
- Voice tips (Windows/Chrome): use headphones, allow mic; if TTS is muted, click Start Camera once to resume audio; Brave/Edge also work.
- If you see server auto-restarts, ensure HF_HOME points outside the project; the backend sets it if unset.

Example curl
------------
curl -F "file=@/path/to/image.jpg" -F "question=What is on the table?" http://127.0.0.1:8000/analyze

Directory Structure
-------------------
backend.py                 # FastAPI, YOLO+BLIP pipeline
frontend/                  # Static UI (served at /ui)
  index.html
  styles.css
  app.js
data/                      # Datasets (created by notebook)
models/                    # Local models if any
.cache/                    # App-local cache (non-HF)

Environment Variables (optional)
--------------------------------
- YOLO_VARIANT: e.g. yolov8n.pt (default), yolov8s.pt, yolov8m.pt
- CAPTION_CKPT: e.g. Salesforce/blip-image-captioning-base (default), large variants
- VQA_CKPT: e.g. Salesforce/blip-vqa-base (default)
- REASONER_CKPT: e.g. google/flan-t5-small (default). Set to empty/invalid to disable.

Model comparison & metrics
--------------------------
- To compare YOLO sizes, set YOLO_VARIANT and evaluate in backend.ipynb.
- Ultralytics writes charts to runs/detect/val_*/: PR_curve.png, F1_curve.png, results.png, conf_matrix.png.


