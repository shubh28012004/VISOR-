# Fusion Evaluation Metrics

## Overview

We now have numerical evaluation metrics to quantitatively assess fusion performance compared to individual model outputs.

## Available Metrics

### 1. **BLEU Scores (BLEU-1, BLEU-2, BLEU-3, BLEU-4)**
- Measures n-gram overlap between generated text and reference
- BLEU-1: Unigram precision
- BLEU-4: 4-gram precision (standard for captioning)
- Range: 0.0 to 1.0 (higher is better)

### 2. **ROUGE-L**
- Measures longest common subsequence (LCS) overlap
- Captures sentence-level structure similarity
- Range: 0.0 to 1.0 (higher is better)

### 3. **METEOR**
- Synonym-aware matching using WordNet
- More semantic than BLEU/ROUGE
- Range: 0.0 to 1.0 (higher is better)

### 4. **Object Coverage**
- Percentage of detected objects mentioned in narrative/caption
- Measures how well detections are incorporated into text output
- Range: 0.0 to 1.0 (higher is better)

## How to Run Evaluation

### Quick Evaluation (20 images)
```bash
python scripts/quick_fusion_eval.py
```

### Full Evaluation (custom number of images)
```bash
python scripts/evaluate_fusion.py \
  --num_images 50 \
  --image_dir data/coco/images/val2017 \
  --captions_json data/coco/annotations/captions_val2017.json \
  --output run/fusion_evaluation_results.json
```

## Expected Output

The evaluation compares three outputs:
1. **BLIP Caption Alone**: Raw caption from BLIP
2. **Fused Narrative**: Combined YOLO + BLIP via reasoner
3. **YOLO Detections (text)**: Detections formatted as text (baseline)

### Example Results Format

```
Metric               BLIP Caption       Fused Narrative    Detections
BLEU-1               0.5234             0.6123             0.2341
BLEU-4               0.3456             0.4123             0.1234
ROUGE-L              0.4789             0.5567             0.2876
METEOR               0.4234             0.5123             0.2345
object_coverage      0.4567             0.8234             1.0000
```

## Interpretation

- **Fused Narrative should show**:
  - Higher BLEU/ROUGE-L/METEOR than BLIP alone (better semantic match)
  - Higher object_coverage than BLIP alone (better integration of detections)
  - Better balance than detections alone (more natural language)

## Requirements

```bash
pip install nltk rouge-score
```

The script will auto-download NLTK data on first run.

## Notes

- Evaluation uses COCO validation captions as ground truth
- If ground truth unavailable, BLIP caption is used as proxy (imperfect but allows comparison)
- Metrics are averaged across all evaluated images
- Full results JSON includes per-image breakdowns

