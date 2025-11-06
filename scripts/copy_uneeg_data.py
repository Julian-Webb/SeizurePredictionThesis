import shutil
import time
from pathlib import Path
from concurrent.futures import as_completed, ThreadPoolExecutor

from data_cleaning.file_correction import MAC_PATTERNS


def copy_item(src: Path, dst: Path):
    """Copy a single folder/file with error handling and progress tracking"""
    try:
        start = time.time()

        if src.is_file():
            # create the parent dir if it doesn't exist
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        else:
            shutil.copytree(src, dst, dirs_exist_ok=True,
                            copy_function=shutil.copy2,
                            ignore_dangling_symlinks=True)
            # Ensure the directory timestamp is preserved
            shutil.copystat(src, dst)

        elapsed = time.time() - start
        print(f"✓ Copied {dst.name} in {elapsed:.1f}s")
        return dst.name, elapsed, True
    except Exception as e:
        print(f"✗ Failed to copy {dst.name}: {e}")
        return dst.name, 0, False


def get_size(path):
    """Get approximate size for progress estimation for both files and folders"""
    try:
        if path.is_file():
            return path.stat().st_size
        else:
            # For folders, recursively sum up all file sizes
            return sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
    except:
        return 0


def main():
    # ||||||||||||||||||||||||||||||||||||||||||||||||||||
    # src_base = Path("/data/datasets")
    src_base = Path("/data/home/webb/original_UNEEG_data")
    dst_base = Path("/data/home/webb/UNEEG_data_1")
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    # all items in these folders will be copied
    parent_folders = [
        "20240201_UNEEG_ForMayo",
        "20250217_UNEEG_Extended",
        "20250501_SUBQ_SeizurePredictionCompetition_2025final/TrainingData"
    ]

    items = []
    for parent_folder in parent_folders:
        for item in (src_base / parent_folder).iterdir():
            # ignore MAC files
            if not (item.name.startswith('._') or item.name in MAC_PATTERNS):
                items.append(Path(parent_folder, item.name))

    if dst_base.exists() and input(f'Remove existing directory {dst_base}? (y/n) ') == 'y':
        print("Removing existing directory...")
        shutil.rmtree(dst_base)

    # Quick size estimation for progress tracking
    print("Estimating copy sizes...")
    total_size = 0
    for item in items:
        src = src_base / item
        size = get_size(src)
        total_size += size
        print(f"  {item}: ~{size / 1e9:.1f} GB")

    print(f"Total: ~{total_size / 1e9:.1f} GB\n")

    start_time = time.time()

    # Create destination directory
    dst_base.mkdir(parents=True, exist_ok=True)

    # Parallel copying
    with ThreadPoolExecutor(max_workers=min(6, len(items))) as executor:
        # Submit all copy jobs
        future_to_item = {}
        for item in items:
            src = src_base / item
            dst = dst_base / item

            future = executor.submit(copy_item, src, dst)
            future_to_item[future] = item

        # Process completed jobs
        completed = 0
        total_jobs = len(future_to_item)

        for future in as_completed(future_to_item):
            folder_name, elapsed, success = future.result()
            completed += 1

            if success:
                progress = (completed / total_jobs) * 100
                print(f"Progress: {completed}/{total_jobs} ({progress:.0f}%)")

    total_time = time.time() - start_time
    print(f"\n🎉 Finished in {total_time:.1f} seconds")
    print(f"Average speed: ~{(total_size / 1e9) / total_time:.1f} GB/s")


if __name__ == "__main__":
    main()
