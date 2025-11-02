#!/usr/bin/env python3
"""
Quick fusion evaluation using test images or sample COCO images.
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from evaluate_fusion import evaluate_fusion, print_summary
import json

# Quick test with available images
image_dir = PROJECT_ROOT / "data" / "coco" / "images" / "val2017"
captions_json = PROJECT_ROOT / "data" / "coco" / "annotations" / "captions_val2017.json"

if image_dir.exists() and image_dir.is_dir():
    # Limit to first 20 images for quick eval
    print("Running quick fusion evaluation...")
    results = evaluate_fusion(
        num_images=20,
        image_dir=image_dir,
        captions_json=captions_json if captions_json.exists() else None
    )
    
    if results:
        print_summary(results['summary'])
        
        # Save to run/ directory
        output_path = PROJECT_ROOT / "run" / "fusion_evaluation_results.json"
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nFull results saved to: {output_path}")
else:
    print(f"COCO val images not found at {image_dir}")
    print("To run full evaluation, use:")
    print("  python scripts/evaluate_fusion.py --num_images 50 --image_dir /path/to/images")

