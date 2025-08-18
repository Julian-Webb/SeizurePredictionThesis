import shutil
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed


def copy_folder(src, dst, folder_name):
    """Copy a single folder with error handling and progress tracking"""
    try:
        start = time.time()
        # Use copy2 to preserve metadata, ignore_dangling_symlinks for robustness
        shutil.copytree(src, dst, dirs_exist_ok=True,
                        copy_function=shutil.copy2,
                        ignore_dangling_symlinks=True)
        elapsed = time.time() - start
        print(f"✓ Copied {folder_name} in {elapsed:.1f}s")
        return folder_name, elapsed, True
    except Exception as e:
        print(f"✗ Failed to copy {folder_name}: {e}")
        return folder_name, 0, False


def get_folder_size(path):
    """Get approximate folder size for progress estimation"""
    try:
        return sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
    except:
        return 0


if __name__ == "__main__":
    src_base = Path("/data/datasets")
    dst_base = Path("/data/home/webb/UNEEG_data_3")

    folders = [
        "20240201_UNEEG_ForMayo",
        "20250217_UNEEG_Extended",
        "20250501_SUBQ_SeizurePredictionCompetition_2025final"
    ]

    if dst_base.exists() and input(f'Remove existing directory {dst_base}? (y/n) ') == 'y':
        print("Removing existing directory...")
        shutil.rmtree(dst_base)

    # Create destination directory
    dst_base.mkdir(parents=True, exist_ok=True)

    # Quick size estimation for progress tracking
    print("Estimating copy sizes...")
    total_size = 0
    for folder in folders:
        src = src_base / folder
        if src.exists():
            size = get_folder_size(src)
            total_size += size
            print(f"  {folder}: ~{size / 1e9:.1f} GB")
        else:
            print(f"  Warning: {folder} not found in source")

    print(f"Total: ~{total_size / 1e9:.1f} GB\n")

    start_time = time.time()

    # Parallel copying with ThreadPoolExecutor
    # Use max 3 workers to avoid overwhelming the I/O system
    with ThreadPoolExecutor(max_workers=min(3, len(folders))) as executor:
        # Submit all copy jobs
        future_to_folder = {}
        for folder in folders:
            src = src_base / folder
            dst = dst_base / folder

            if src.exists():
                future = executor.submit(copy_folder, src, dst, folder)
                future_to_folder[future] = folder
            else:
                print(f"Skipping {folder} - source not found")

        # Process completed jobs
        completed = 0
        total_jobs = len(future_to_folder)

        for future in as_completed(future_to_folder):
            folder_name, elapsed, success = future.result()
            completed += 1

            if success:
                progress = (completed / total_jobs) * 100
                print(f"Progress: {completed}/{total_jobs} ({progress:.0f}%)")

    total_time = time.time() - start_time
    print(f"\n🎉 Finished in {total_time:.1f} seconds")
    print(f"Average speed: ~{(total_size / 1e9) / total_time:.1f} GB/s")