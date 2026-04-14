"""
Comprehensive dataset validation for paddy disease classification dataset.
"""
import os
import sys
from pathlib import Path
from collections import defaultdict, Counter
from PIL import Image
import hashlib

DATA_DIR = Path("/home/revan/Desktop/PR/data/raw/train_images")
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp', '.gif'}

def main():
    class_dirs = sorted([d for d in DATA_DIR.iterdir() if d.is_dir()])
    
    # Accumulators
    class_counts = {}
    corrupted = []
    non_images = []
    size_stats = defaultdict(list)
    too_small = []
    too_large = []
    odd_aspect = []
    file_sizes = defaultdict(list)
    total_images = 0
    
    print("=" * 70)
    print("PADDY DISEASE DATASET VALIDATION REPORT")
    print("=" * 70)
    
    # 1. Class folders and image counts
    print("\n[1] CLASS FOLDERS AND IMAGE COUNTS")
    print("-" * 50)
    
    for cls_dir in class_dirs:
        files = list(cls_dir.iterdir())
        img_files = [f for f in files if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS]
        non_img = [f for f in files if f.is_file() and f.suffix.lower() not in IMAGE_EXTENSIONS]
        non_images.extend(non_img)
        class_counts[cls_dir.name] = len(img_files)
        total_images += len(img_files)
        print(f"  {cls_dir.name:30s} : {len(img_files):5d} images", end="")
        if non_img:
            print(f"  (+{len(non_img)} non-image files)", end="")
        print()
    
    print(f"\n  {'TOTAL':30s} : {total_images:5d} images")
    
    # 2 & 3. Corrupted images and size checks
    print("\n[2] CHECKING EVERY IMAGE FOR CORRUPTION AND SIZE ISSUES...")
    print("    (opening and verifying each file with PIL)")
    
    checked = 0
    for cls_dir in class_dirs:
        for fpath in sorted(cls_dir.iterdir()):
            if not fpath.is_file():
                continue
            if fpath.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            
            checked += 1
            if checked % 2000 == 0:
                print(f"    ... checked {checked}/{total_images} images", file=sys.stderr)
            
            fsize = fpath.stat().st_size
            file_sizes[fsize].append(fpath)
            
            try:
                with Image.open(fpath) as img:
                    img.verify()
                with Image.open(fpath) as img:
                    w, h = img.size
                    size_stats[cls_dir.name].append((w, h))
                    
                    if w < 32 or h < 32:
                        too_small.append((fpath, w, h))
                    if w > 4096 or h > 4096:
                        too_large.append((fpath, w, h))
                    ratio = max(w, h) / max(min(w, h), 1)
                    if ratio > 3.0:
                        odd_aspect.append((fpath, w, h, ratio))
            except Exception as e:
                corrupted.append((fpath, str(e)))
    
    print(f"    Checked {checked} images total.\n")
    
    if corrupted:
        print(f"  CORRUPTED/UNREADABLE IMAGES: {len(corrupted)}")
        for fpath, err in corrupted:
            print(f"    - {fpath.relative_to(DATA_DIR)}  |  Error: {err}")
    else:
        print("  CORRUPTED/UNREADABLE IMAGES: 0  (all images OK)")
    
    # 3. Size analysis
    print("\n[3] IMAGE SIZE ANALYSIS")
    print("-" * 50)
    
    all_sizes = []
    for cls, sizes in size_stats.items():
        all_sizes.extend(sizes)
    
    if all_sizes:
        widths = [s[0] for s in all_sizes]
        heights = [s[1] for s in all_sizes]
        size_counter = Counter(all_sizes)
        
        print(f"  Width  range: {min(widths)} - {max(widths)}")
        print(f"  Height range: {min(heights)} - {max(heights)}")
        print(f"  Most common sizes (top 5):")
        for (w, h), cnt in size_counter.most_common(5):
            print(f"    {w}x{h} : {cnt} images ({100*cnt/len(all_sizes):.1f}%)")
        
        if too_small:
            print(f"\n  TOO SMALL (<32px): {len(too_small)}")
            for fpath, w, h in too_small[:10]:
                print(f"    - {fpath.relative_to(DATA_DIR)} ({w}x{h})")
        else:
            print(f"\n  TOO SMALL (<32px): 0")
        
        if too_large:
            print(f"  TOO LARGE (>4096px): {len(too_large)}")
            for fpath, w, h in too_large[:10]:
                print(f"    - {fpath.relative_to(DATA_DIR)} ({w}x{h})")
        else:
            print(f"  TOO LARGE (>4096px): 0")
        
        if odd_aspect:
            print(f"  ODD ASPECT RATIO (>3:1): {len(odd_aspect)}")
            for fpath, w, h, r in odd_aspect[:10]:
                print(f"    - {fpath.relative_to(DATA_DIR)} ({w}x{h}, ratio={r:.1f})")
        else:
            print(f"  ODD ASPECT RATIO (>3:1): 0")
    
    # 4. Duplicate detection
    print("\n[4] DUPLICATE FILE DETECTION (by file size + MD5)")
    print("-" * 50)
    
    potential_dupes = {sz: paths for sz, paths in file_sizes.items() if len(paths) > 1}
    
    exact_dupes = []
    for sz, paths in potential_dupes.items():
        hash_map = defaultdict(list)
        for p in paths:
            h = hashlib.md5(p.read_bytes()).hexdigest()
            hash_map[h].append(p)
        for h, group in hash_map.items():
            if len(group) > 1:
                exact_dupes.append(group)
    
    if exact_dupes:
        print(f"  EXACT DUPLICATE GROUPS (same MD5): {len(exact_dupes)}")
        for i, group in enumerate(exact_dupes[:15], 1):
            print(f"  Group {i} ({len(group)} files):")
            for p in group:
                print(f"    - {p.relative_to(DATA_DIR)}")
    else:
        print("  EXACT DUPLICATES: 0  (no identical files found)")
    
    same_size_count = sum(len(paths) for paths in potential_dupes.values())
    print(f"  Files sharing a file size with at least one other: {same_size_count} across {len(potential_dupes)} size groups")
    
    # 5. Non-image files
    print("\n[5] NON-IMAGE FILES")
    print("-" * 50)
    if non_images:
        print(f"  Found {len(non_images)} non-image files:")
        for f in non_images:
            print(f"    - {f.relative_to(DATA_DIR)} (ext: {f.suffix})")
    else:
        print("  None found. All files have recognized image extensions.")
    
    # 6. Class imbalance
    print("\n[6] CLASS IMBALANCE ANALYSIS")
    print("-" * 50)
    if class_counts:
        counts = list(class_counts.values())
        max_count = max(counts)
        min_count = min(counts)
        mean_count = sum(counts) / len(counts)
        
        max_class = [k for k, v in class_counts.items() if v == max_count][0]
        min_class = [k for k, v in class_counts.items() if v == min_count][0]
        
        imbalance_ratio = max_count / max(min_count, 1)
        
        print(f"  Largest class:  {max_class} ({max_count})")
        print(f"  Smallest class: {min_class} ({min_count})")
        print(f"  Mean count:     {mean_count:.1f}")
        print(f"  Imbalance ratio (max/min): {imbalance_ratio:.2f}x")
        print()
        print(f"  {'Class':30s}  {'Count':>6s}  {'Ratio to Max':>12s}  {'Bar'}")
        print(f"  {'-'*30}  {'-'*6}  {'-'*12}  {'-'*30}")
        for cls in sorted(class_counts, key=class_counts.get, reverse=True):
            cnt = class_counts[cls]
            ratio = cnt / max_count
            bar = '#' * int(ratio * 30)
            print(f"  {cls:30s}  {cnt:6d}  {ratio:12.2%}  {bar}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Total classes:        {len(class_counts)}")
    print(f"  Total images:         {total_images}")
    print(f"  Corrupted images:     {len(corrupted)}")
    print(f"  Too small (<32px):    {len(too_small)}")
    print(f"  Too large (>4096px):  {len(too_large)}")
    print(f"  Odd aspect ratio:     {len(odd_aspect)}")
    print(f"  Exact duplicates:     {sum(len(g) for g in exact_dupes)} files in {len(exact_dupes)} groups")
    print(f"  Non-image files:      {len(non_images)}")
    print(f"  Imbalance ratio:      {max(class_counts.values())/max(min(class_counts.values()),1):.2f}x")
    
    issues = len(corrupted) + len(too_small) + len(too_large) + len(odd_aspect) + len(non_images) + len(exact_dupes)
    if issues == 0:
        print("\n  STATUS: Dataset looks clean. No issues detected.")
    else:
        print(f"\n  STATUS: {issues} issue(s) detected. Review above for details.")
    print("=" * 70)

if __name__ == "__main__":
    main()
