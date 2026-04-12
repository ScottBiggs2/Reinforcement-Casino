#!/usr/bin/env python3
import os
import torch
import glob
import argparse
from tqdm import tqdm

def convert_pt_to_bool(path, dry_run=False):
    """Loads a mask .pt file, converts tensors to bool, and saves back."""
    size_old = os.path.getsize(path) / (1024**3)
    if size_old < 10:
        # If it's already small, likely already bool or bitpacked
        print(f"Skipping {os.path.basename(path)} (already small: {size_old:.2f} GB)")
        return

    print(f"Loading {os.path.basename(path)} ({size_old:.2f} GB)...")
    if dry_run:
        return

    try:
        # Load on CPU to avoid login node OOM
        data = torch.load(path, map_location='cpu', weights_only=False)
        
        # Handle wrapped format {"masks": {...}, "metadata": {...}}
        is_wrapped = isinstance(data, dict) and "masks" in data
        masks = data["masks"] if is_wrapped else data
        
        if not isinstance(masks, dict):
            print(f"  Error: data in {path} is not a dictionary of masks.")
            return

        print(f"  Converting {len(masks)} mask tensors to bool...")
        for name in masks:
            masks[name] = masks[name].to(torch.bool)
            
        print(f"  Saving...")
        torch.save(data, path)
        size_new = os.path.getsize(path) / (1024**3)
        print(f"  ✓ Success. New size: {size_new:.2f} GB (Saved ~{size_old - size_new:.1f} GB)")
    except Exception as e:
        print(f"  FAILED to convert {path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert masks from float32 to bool to save space.")
    parser.add_argument("--dir", type=str, required=True, help="Directory containing .pt masks")
    parser.add_argument("--dry_run", action="store_true", help="Only show what would be done")
    args = parser.parse_args()

    # Search for all .pt files in the directory
    files = glob.glob(os.path.join(args.dir, "*.pt"))
    if not files:
        print(f"No .pt files found in {args.dir}")
    else:
        print(f"Found {len(files)} potential mask files.")
        for f in tqdm(files):
            convert_pt_to_bool(f, dry_run=args.dry_run)
