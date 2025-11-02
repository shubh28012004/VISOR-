#!/usr/bin/env python3
"""
Evaluate Fusion Pipeline: Compare fused narratives vs. individual model outputs.

Metrics computed:
- BLEU-1/2/3/4: n-gram overlap
- ROUGE-L: Longest common subsequence
- METEOR: Synonym-aware matching
- Semantic similarity (if available)
- Object coverage: How many detected objects are mentioned in narrative
"""

import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import json
import time
from typing import List, Dict, Any
from collections import defaultdict

# Metrics libraries
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.translate.meteor_score import meteor_score
    import nltk
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
except ImportError:
    print("Installing nltk...")
    os.system("pip install nltk rouge-score")
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.translate.meteor_score import meteor_score
    import nltk
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)

try:
    from rouge_score import rouge_scorer
except ImportError:
    rouge_scorer = None

from PIL import Image
from backend import caption, detect, generate_narrative, analyze_image

# Ensure transformers uses PyTorch only
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"


def format_detections_as_text(detections: List[Dict[str, Any]], max_items: int = 6) -> str:
    """Convert YOLO detections to a simple text format."""
    items = []
    for d in detections[:max_items]:
        name = d.get('class_name', f"object_{d.get('class_id', '?')}")
        conf = d.get('confidence', 0.0)
        items.append(f"{name} {conf*100:.0f}%")
    return ", ".join(items) if items else "no objects detected"


def compute_bleu(reference: str, candidate: str) -> Dict[str, float]:
    """Compute BLEU-1, BLEU-2, BLEU-3, BLEU-4 scores."""
    ref_tokens = reference.lower().split()
    cand_tokens = candidate.lower().split()
    smoothing = SmoothingFunction().method1
    
    scores = {}
    for n in range(1, 5):
        weights = tuple([1.0/n] * n + [0.0] * (4-n))
        try:
            scores[f'BLEU-{n}'] = sentence_bleu([ref_tokens], cand_tokens, weights=weights, smoothing_function=smoothing)
        except:
            scores[f'BLEU-{n}'] = 0.0
    return scores


def compute_rouge_l(reference: str, candidate: str) -> float:
    """Compute ROUGE-L score."""
    if rouge_scorer is None:
        # Simple fallback: compute longest common subsequence ratio
        ref_words = reference.lower().split()
        cand_words = candidate.lower().split()
        
        def lcs_length(x, y):
            m, n = len(x), len(y)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if x[i-1] == y[j-1]:
                        dp[i][j] = dp[i-1][j-1] + 1
                    else:
                        dp[i][j] = max(dp[i-1][j], dp[i][j-1])
            return dp[m][n]
        
        lcs = lcs_length(ref_words, cand_words)
        if len(ref_words) == 0 or len(cand_words) == 0:
            return 0.0
        precision = lcs / len(cand_words) if len(cand_words) > 0 else 0.0
        recall = lcs / len(ref_words) if len(ref_words) > 0 else 0.0
        if precision + recall == 0:
            return 0.0
        return 2.0 * precision * recall / (precision + recall)
    else:
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        scores = scorer.score(reference, candidate)
        return scores['rougeL'].fmeasure


def compute_meteor(reference: str, candidate: str) -> float:
    """Compute METEOR score."""
    try:
        return meteor_score([reference.split()], candidate.split())
    except:
        return 0.0


def compute_object_coverage(detections: List[Dict[str, Any]], text: str) -> Dict[str, float]:
    """Compute how many detected objects are mentioned in the text."""
    if not detections:
        return {'coverage': 0.0, 'mentioned': 0, 'total': 0}
    
    text_lower = text.lower()
    mentioned = 0
    for d in detections:
        name = d.get('class_name', '').lower()
        if name and name in text_lower:
            mentioned += 1
    
    total = len(detections)
    coverage = mentioned / total if total > 0 else 0.0
    
    return {
        'coverage': coverage,
        'mentioned': mentioned,
        'total': total
    }


def evaluate_on_image(image_path: Path, ground_truth_caption: str = None) -> Dict[str, Any]:
    """Evaluate all outputs for a single image."""
    result = analyze_image(image_path)
    
    blip_caption = result['caption']
    detections = result['detections']
    fused_narrative = result.get('narrative', '')
    
    # Format detections as text
    detections_text = format_detections_as_text(detections)
    
    # If no ground truth, use BLIP caption as proxy (imperfect but allows comparison)
    if ground_truth_caption is None:
        ground_truth_caption = blip_caption
    
    results = {
        'image': str(image_path.name),
        'ground_truth': ground_truth_caption,
        'blip_caption': blip_caption,
        'detections_text': detections_text,
        'fused_narrative': fused_narrative,
        'num_detections': len(detections),
    }
    
    # Compute metrics for BLIP caption
    if blip_caption:
        results['blip_metrics'] = {
            **compute_bleu(ground_truth_caption, blip_caption),
            'ROUGE-L': compute_rouge_l(ground_truth_caption, blip_caption),
            'METEOR': compute_meteor(ground_truth_caption, blip_caption),
        }
        results['blip_metrics']['object_coverage'] = compute_object_coverage(detections, blip_caption)['coverage']
    else:
        results['blip_metrics'] = {}
    
    # Compute metrics for fused narrative
    if fused_narrative:
        results['fused_metrics'] = {
            **compute_bleu(ground_truth_caption, fused_narrative),
            'ROUGE-L': compute_rouge_l(ground_truth_caption, fused_narrative),
            'METEOR': compute_meteor(ground_truth_caption, fused_narrative),
        }
        results['fused_metrics']['object_coverage'] = compute_object_coverage(detections, fused_narrative)['coverage']
    else:
        results['fused_metrics'] = {}
    
    # Compute metrics for detections text (baseline)
    if detections_text:
        results['detections_metrics'] = {
            **compute_bleu(ground_truth_caption, detections_text),
            'ROUGE-L': compute_rouge_l(ground_truth_caption, detections_text),
            'METEOR': compute_meteor(ground_truth_caption, detections_text),
        }
        results['detections_metrics']['object_coverage'] = 1.0  # By definition
    else:
        results['detections_metrics'] = {}
    
    return results


def evaluate_fusion(num_images: int = 50, image_dir: Path = None, captions_json: Path = None):
    """Evaluate fusion pipeline on a set of images."""
    if image_dir is None:
        # Try to find COCO val images
        coco_dir = PROJECT_ROOT / "data" / "coco" / "images" / "val2017"
        if coco_dir.exists():
            image_dir = coco_dir
            print(f"Using COCO val images from: {image_dir}")
        else:
            print("ERROR: No image directory specified and COCO val not found.")
            print("Usage: evaluate_fusion(num_images=50, image_dir=Path('path/to/images'), captions_json=Path('path/to/captions.json'))")
            return None
    
    # Load captions if available
    captions_map = {}
    if captions_json and captions_json.exists():
        with open(captions_json) as f:
            data = json.load(f)
            # Assume COCO format
            if 'annotations' in data:
                for ann in data['annotations']:
                    img_id = ann['image_id']
                    if img_id not in captions_map:
                        captions_map[img_id] = []
                    captions_map[img_id].append(ann['caption'])
    
    # Get image files
    image_files = list(image_dir.glob("*.jpg"))[:num_images]
    if not image_files:
        image_files = list(image_dir.glob("*.png"))[:num_images]
    
    print(f"Evaluating on {len(image_files)} images...")
    
    all_results = []
    for i, img_path in enumerate(image_files):
        print(f"Processing {i+1}/{len(image_files)}: {img_path.name}", end='\r')
        
        # Try to get ground truth caption
        gt_caption = None
        if captions_map:
            # Simple heuristic: use image filename as key (may need adjustment for COCO)
            img_id = img_path.stem
            if img_id in captions_map:
                gt_caption = captions_map[img_id][0]  # Use first caption
        
        result = evaluate_on_image(img_path, gt_caption)
        all_results.append(result)
    
    print(f"\nCompleted evaluation on {len(all_results)} images.")
    
    # Aggregate metrics
    metrics = defaultdict(list)
    for result in all_results:
        for method in ['blip_metrics', 'fused_metrics', 'detections_metrics']:
            if method in result and result[method]:
                for metric_name, value in result[method].items():
                    metrics[f"{method}_{metric_name}"].append(value)
    
    # Compute averages
    summary = {}
    for key, values in metrics.items():
        if values:
            summary[key] = {
                'mean': sum(values) / len(values),
                'std': (sum((x - sum(values)/len(values))**2 for x in values) / len(values))**0.5 if len(values) > 1 else 0.0,
                'count': len(values)
            }
    
    return {
        'summary': summary,
        'detailed_results': all_results
    }


def print_summary(summary: Dict[str, Any]):
    """Print formatted summary of evaluation results."""
    print("\n" + "="*80)
    print("FUSION EVALUATION SUMMARY")
    print("="*80)
    
    # Extract method names and metrics
    methods = {
        'blip_metrics': 'BLIP Caption',
        'fused_metrics': 'Fused Narrative',
        'detections_metrics': 'YOLO Detections (text)'
    }
    
    metric_names = ['BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4', 'ROUGE-L', 'METEOR', 'object_coverage']
    
    # Print header
    print(f"\n{'Metric':<20} {'BLIP Caption':<20} {'Fused Narrative':<20} {'Detections':<20}")
    print("-"*80)
    
    for metric in metric_names:
        row = [metric]
        for method_key, method_label in methods.items():
            key = f"{method_key}_{metric}"
            if key in summary:
                mean = summary[key]['mean']
                row.append(f"{mean:.4f}")
            else:
                row.append("N/A")
        print(f"{row[0]:<20} {row[1]:<20} {row[2]:<20} {row[3]:<20}")
    
    print("\n" + "="*80)
    print("Key Findings:")
    
    # Compare fused vs BLIP
    for metric in ['BLEU-4', 'ROUGE-L', 'METEOR', 'object_coverage']:
        blip_key = f"blip_metrics_{metric}"
        fused_key = f"fused_metrics_{metric}"
        
        if blip_key in summary and fused_key in summary:
            blip_val = summary[blip_key]['mean']
            fused_val = summary[fused_key]['mean']
            improvement = ((fused_val - blip_val) / blip_val * 100) if blip_val > 0 else 0
            
            if improvement > 0:
                print(f"  ✓ {metric}: Fused (+{improvement:.1f}% vs BLIP)")
            elif improvement < 0:
                print(f"  - {metric}: Fused ({improvement:.1f}% vs BLIP)")
    
    print("="*80 + "\n")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate fusion pipeline')
    parser.add_argument('--num_images', type=int, default=50, help='Number of images to evaluate')
    parser.add_argument('--image_dir', type=str, help='Directory containing images')
    parser.add_argument('--captions_json', type=str, help='JSON file with ground truth captions')
    parser.add_argument('--output', type=str, help='Output JSON file for results')
    
    args = parser.parse_args()
    
    image_dir = Path(args.image_dir) if args.image_dir else None
    captions_json = Path(args.captions_json) if args.captions_json else None
    
    results = evaluate_fusion(
        num_images=args.num_images,
        image_dir=image_dir,
        captions_json=captions_json
    )
    
    if results:
        print_summary(results['summary'])
        
        if args.output:
            output_path = Path(args.output)
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nDetailed results saved to: {output_path}")

