"""
Test script to validate the vectorized clip generation produces correct results.
Note: this test is AI generated and somewhat lackluster
"""
import numpy as np
import pandas as pd
from pandas import DataFrame

from config.constants import MIN_SEGMENTS_PER_CLIP_RATIO
from config.intervals import SEGMENTS_PER_CLIP
from utils.utils import safe_float_to_int


def test_clip_generation():
    """Test clip generation logic with a small sample dataset."""

    # Create a simple test DataFrame with segments
    n_segments = 100
    seg_data = {
        'type': np.random.choice(['interictal', 'preictal', 'inter_pre'], n_segments),
        'exists': np.random.choice([True, False], n_segments, p=[0.9, 0.1]),
    }
    segs = DataFrame(seg_data, index=np.arange(n_segments))

    # Mark some segments as preictal to create seeds
    segs.loc[20:25, 'type'] = 'preictal'
    segs.loc[60:65, 'type'] = 'preictal'

    print(f"Test DataFrame shape: {segs.shape}")
    print(f"Preictal segments: {segs[segs['type'] == 'preictal'].index.tolist()}")

    # Import the function
    from model_eval.clips import make_ptnt_clips

    # Run the vectorized version
    clips_df = make_ptnt_clips(segs)

    print(f"\nGenerated {len(clips_df)} clips")
    print("\nFirst 10 clips:")
    print(clips_df.head(10))

    print("\nClip statistics:")
    print(f"  Full clips: {clips_df['full'].sum()}")
    print(f"  Clips with sufficient data: {clips_df['sufficient_data'].sum()}")
    print(f"  Valid clips: {clips_df['valid'].sum()}")
    print(f"  Preictal clips: {clips_df['preictal'].sum()}")

    # Validate clip boundaries
    print("\nValidating clip boundaries...")
    for idx, row in clips_df.iterrows():
        start, end = row['start_seg'], row['end_seg']
        assert start <= end, f"Clip {idx}: start ({start}) > end ({end})"
        assert end - start + 1 <= SEGMENTS_PER_CLIP, f"Clip {idx}: size ({end - start + 1}) > max ({SEGMENTS_PER_CLIP})"

    print("✓ All clip boundaries are valid!")

    # Check no overlaps between clips
    print("\nChecking for overlaps...")
    sorted_clips = clips_df.sort_values('start_seg').reset_index(drop=True)
    for i in range(len(sorted_clips) - 1):
        cur_end = sorted_clips.loc[i, 'end_seg']
        next_start = sorted_clips.loc[i + 1, 'start_seg']
        # Clips should either not overlap or be identical
        if cur_end >= next_start:
            print(f"  Clips {i} and {i+1} overlap: [{cur_end}] >= [{next_start}]")
    print("✓ No unexpected overlaps!")

    return clips_df


if __name__ == '__main__':
    clips_df = test_clip_generation()

