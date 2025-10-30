import os
from pathlib import Path
from typing import Dict, Any, Optional, List

# Ensure PyTorch-only for transformers
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
"""
Important: keep the Hugging Face cache outside the project directory to
avoid uvicorn StatReload watching large model download updates and
restarting the server mid-run.
"""
_hf_home = os.environ.get("HF_HOME")
if not _hf_home:
    import pathlib as _p
    os.environ["HF_HOME"] = str(_p.Path.home() / ".cache" / "huggingface")
    os.environ.setdefault("TRANSFORMERS_CACHE", os.environ["HF_HOME"]) 

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import tempfile

from PIL import Image
import torch
from ultralytics import YOLO
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    BlipForQuestionAnswering,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)

# Directories
PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
CACHE_DIR = PROJECT_ROOT / ".cache"  # app-local misc cache (not HF cache)
FRONTEND_DIR = PROJECT_ROOT / "frontend"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Model config (SLM-first)
# Allow overriding via environment variables for accuracy/speed tradeoffs
YOLO_VARIANT = os.environ.get("YOLO_VARIANT", "yolov8n.pt")
CAPTION_CKPT = os.environ.get("CAPTION_CKPT", "Salesforce/blip-image-captioning-base")
VQA_CKPT = os.environ.get("VQA_CKPT", "Salesforce/blip-vqa-base")

# Small language model for scene reasoning (generates richer narrative)
REASONER_CKPT = os.environ.get("REASONER_CKPT", "google/flan-t5-small")

# Device selection
if torch.cuda.is_available():
    DEVICE = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

# Load models
_yolo: Optional[YOLO] = None
_caption_processor: Optional[BlipProcessor] = None
_caption_model: Optional[BlipForConditionalGeneration] = None
_vqa_processor: Optional[BlipProcessor] = None
_vqa_model: Optional[BlipForQuestionAnswering] = None
_reasoner_tokenizer: Optional[AutoTokenizer] = None
_reasoner_model: Optional[AutoModelForSeq2SeqLM] = None


def _lazy_load():
    global _yolo, _caption_processor, _caption_model, _vqa_processor, _vqa_model, _reasoner_tokenizer, _reasoner_model
    if _yolo is None:
        _yolo = YOLO(YOLO_VARIANT)
    if _caption_processor is None:
        # Use HF_HOME/TRANSFORMERS_CACHE; avoid writing to project .cache to prevent reload loops
        _caption_processor = BlipProcessor.from_pretrained(CAPTION_CKPT)
    if _caption_model is None:
        _caption_model = BlipForConditionalGeneration.from_pretrained(CAPTION_CKPT).to(DEVICE)
    if _vqa_processor is None:
        _vqa_processor = BlipProcessor.from_pretrained(VQA_CKPT)
    if _vqa_model is None:
        _vqa_model = BlipForQuestionAnswering.from_pretrained(VQA_CKPT).to(DEVICE)
    if _reasoner_tokenizer is None or _reasoner_model is None:
        try:
            _reasoner_tokenizer = AutoTokenizer.from_pretrained(REASONER_CKPT)
            _reasoner_model = AutoModelForSeq2SeqLM.from_pretrained(REASONER_CKPT).to(DEVICE)
        except Exception:
            # Reasoner is optional; continue without it
            _reasoner_tokenizer = None
            _reasoner_model = None


def detect(image_path: Path, conf: float = 0.25):
    _lazy_load()
    results = _yolo.predict(source=str(image_path), conf=conf, verbose=False)
    return results[0]


def caption(image_path: Path, max_new_tokens: int = 30) -> str:
    _lazy_load()
    image = Image.open(image_path).convert("RGB")
    inputs = _caption_processor(images=image, return_tensors="pt").to(DEVICE)
    out = _caption_model.generate(**inputs, max_new_tokens=max_new_tokens)
    text = _caption_processor.batch_decode(out, skip_special_tokens=True)[0]
    return text


def vqa(image_path: Path, question: str, max_new_tokens: int = 20) -> str:
    _lazy_load()
    image = Image.open(image_path).convert("RGB")
    inputs = _vqa_processor(images=image, text=question, return_tensors="pt").to(DEVICE)
    out = _vqa_model.generate(**inputs, max_new_tokens=max_new_tokens)
    ans = _vqa_processor.batch_decode(out, skip_special_tokens=True)[0]
    return ans


def generate_narrative(caption_text: str, detections: List[Dict[str, Any]], max_new_tokens: int = 80) -> Optional[str]:
    """Use a small instruction-tuned model to synthesize a concise scene description.

    The prompt fuses the raw caption and the top detections into a short, natural sentence
    optimized for text-to-speech.
    """
    _lazy_load()
    if _reasoner_tokenizer is None or _reasoner_model is None:
        return None

    # Build a compact, structured context
    det_summ = ", ".join(
        f"{d.get('class_name', d.get('class_id'))} {(d.get('confidence', 0.0)*100):.0f}%"
        for d in detections[:6]
    ) or "none"

    instruction = (
        "You are a helpful assistant for blind users. Write one natural sentence "
        "that clearly describes the scene based on the provided vision outputs. "
        "Avoid hallucinations, avoid speculation, and keep it under 25 words."
    )
    prompt = (
        f"Instruction: {instruction}\n"
        f"Caption: {caption_text or 'n/a'}\n"
        f"Detections: {det_summ}\n"
        f"Response:"
    )

    inputs = _reasoner_tokenizer([prompt], return_tensors="pt").to(DEVICE)
    output_ids = _reasoner_model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=2,
        early_stopping=True,
    )
    text = _reasoner_tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    # Some T5-style models may echo the prompt; extract after 'Response:' if present
    if "Response:" in text:
        text = text.split("Response:", 1)[-1].strip()
    return text.strip() or None


def analyze_image(image_path: Path, question: Optional[str] = None) -> Dict[str, Any]:
    det = detect(image_path)
    cap = caption(image_path)
    
    _q = (question or "").strip()
    if not _q:
        _q = "What is in front of me?"
    qa = vqa(image_path, _q)

    names = getattr(det, "names", None) or {}
    det_boxes: List[Dict[str, Any]] = []
    try:
        for b in det.boxes:
            cls_id = int(b.cls[0].item())
            conf = float(b.conf[0].item())
            xyxy = [float(v) for v in b.xyxy[0].tolist()]
            cls_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
            det_boxes.append({
                "class_id": cls_id,
                "class_name": cls_name,
                "confidence": conf,
                "xyxy": xyxy,
            })
    except Exception:
        pass

    narrative = generate_narrative(cap, det_boxes)

    return {
        "device": DEVICE,
        "caption": cap,
        "vqa_answer": qa,
        "detections": det_boxes,
        "narrative": narrative,
        "reasoner_enabled": _reasoner_model is not None,
        "narrative_source": ("reasoner" if narrative else "caption"),
    }


app = FastAPI(title="VISOR Backend (SLM)")
_READY = False

# Dev CORS (open)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static frontend if present
if FRONTEND_DIR.exists():
    app.mount("/ui", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="ui")


@app.on_event("startup")
def _warmup_startup():
    global _READY  # CRITICAL FIX: Declare global to modify module-level variable
    try:
        _lazy_load()
        # Create a proper dummy image: 224x224 RGB (BLIP's expected size)
        from PIL import Image as _I
        import io as _io
        buf = _io.BytesIO()
        _I.new("RGB", (224, 224), (128, 128, 128)).save(buf, format="JPEG")
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(buf.getvalue())
            tmp_path = Path(tmp.name)
        try:
            _ = analyze_image(tmp_path, question="What is in front of me?")
        finally:
            try: 
                tmp_path.unlink(missing_ok=True)
            except Exception: 
                pass
        _READY = True
        print(f"✓ Models loaded successfully on {DEVICE}")
    except Exception as e:
        _READY = False
        print(f"✗ Warmup failed: {e}")


@app.post("/analyze")
async def analyze_endpoint(file: UploadFile = File(...), question: Optional[str] = Form(default=None)):
    if not _READY:
        return JSONResponse(content={"ready": False, "message": "Model warming up"}, status_code=503)
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)
    try:
        result = analyze_image(tmp_path, question=question)
        result["ready"] = True
        return JSONResponse(content=result)
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass


# Health check
@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE, "ready": _READY}