import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from models.load_data import load_data
from feature_extraction.extract_features import FeatureNames
from preprocessing.create_clips import map_segs_to_clips
from utils import QueuedCall, run_queued_calls_on_gpus
from config import PatientDir, PATHS, save_dataframe_multiformat
from utils.utils import FunctionTimer


def calc_seg_scores_for_model(
        segs: DataFrame,
        model_path: Path,
        data_type: Literal['eeg', 'features'],
        edf_dir: Path,
        feature_scaler_path: Path,
        logging_prefix: str = 'unknown ptnt - unknown model',
        batch_size: int = 2 ** 19,
):
    import tensorflow as tf  # Local import, so only GPUs available to this process are initialized

    logging.info(f'[{logging_prefix}] 🎬 Making raw predictions. Loading data...')
    with FunctionTimer(f'load_data for {logging_prefix}'):
        data = load_data(
            segs,
            data_type,
            subsample_shuffle_and_subselect_types=False,
            train=True,
            test=True,
            feature_names=FeatureNames.ENSEMBLE,
            edf_dir=edf_dir,
        )

    if data_type == 'features':
        # Perform z-scaling for features
        with open(feature_scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        data['x'] = scaler.transform(data['x'])

    # Predict in batches to avoid overallocating GPU memory
    model = tf.keras.models.load_model(model_path)
    n_segs = data['x'].shape[0]
    scores = []
    for i in range(0, n_segs, batch_size):
        end = min(i + batch_size, n_segs)
        logging.info(f'[{logging_prefix}] ⏳ Processing segments {i}-{end - 1} of {n_segs}...')

        batch = data['x'][i:end]
        batch_scores = model.predict(batch).reshape(-1) # predict outputs with shape (n, 1)
        scores.append(batch_scores)

    scores = np.concatenate(scores)

    # Associate scores with segments (original index and start time)
    model_scores = Series(scores, index=data['index_and_start'].index)
    return model_scores


def calc_clips_scores_for_ptnt(
        seg_scores: DataFrame,
        clips: DataFrame,
        score_cols: list[str],
):
    """Calculate clip scores for a patient given segment scores."""
    clip_id = map_segs_to_clips(seg_scores.index, clips['start_seg'].values)
    segs_by_clip = seg_scores[score_cols].groupby(clip_id)
    clip_scores = segs_by_clip.mean()  # NA values are skipped by default
    return clip_scores


def calc_scores_for_pdir(
        pdir: PatientDir,
        models: Iterable[str] = ('ensemble', 'CNN'),
):
    """Calculate raw scores (rather than classifications) for a patient for segments and clips."""
    # ---- Calculate segment scores
    specs_per_model = {
        'CNN': (pdir.cnn_model, 'eeg'),
        'ensemble': (pdir.ensemble_model, 'features'),
    }
    segs = pd.read_pickle(pdir.segments_table.pickle)

    seg_scores = segs[['start_mtz']].copy()  # scores per segment and model

    for model in models:
        model_path, data_type = specs_per_model[model]
        # noinspection PyTypeChecker
        model_seg_scores = calc_seg_scores_for_model(segs, model_path, data_type, pdir.edf_dir, pdir.feature_scaler,
                                                     logging_prefix=f'{pdir.name} - {model}')
        seg_scores[f'{model}_score'] = model_seg_scores.reindex(seg_scores.index)

    save_dataframe_multiformat(seg_scores, pdir.segment_scores_table)
    logging.info(f'[{pdir.name}] ⏳ Saved segment scores to {pdir.segment_scores_table}')

    # ---- Calculate clip scores
    clips = pd.read_pickle(pdir.clips_table.pickle)
    score_cols = [f'{model}_score' for model in models]
    clips.loc[:, score_cols] = calc_clips_scores_for_ptnt(seg_scores, clips, score_cols)

    save_dataframe_multiformat(clips, pdir.clip_scores_table)
    logging.info(f'[{pdir.name}] ✅ Saved clip scores to {pdir.clip_scores_table}')


def calc_scores_for_pdirs(
        pdirs: list[PatientDir],
        serial_processing: bool = False,
        available_gpus: list[int] = None,
        merged_log_file: Path = None,
) -> Path | None:
    if merged_log_file is not None:
        merged_log_file.parent.mkdir(parents=True, exist_ok=True)
        logging.info("Writing merged model-eval log to %s", merged_log_file)

    if serial_processing:
        for pdir in pdirs:
            calc_scores_for_pdir(pdir)

    else:
        run_log_dir = PATHS.logs_dir / f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_calc_seg_scores_gpu_logs"
        logging.info("Writing per-GPU seg scores logs to %s", run_log_dir)

        tasks = [
            QueuedCall(func=calc_scores_for_pdir, args=(pdir,), label=f'{pdir.name} - calc_seg_scores')
            for pdir in pdirs]

        run_queued_calls_on_gpus(
            tasks=tasks,
            gpus=available_gpus,
            log_dir=run_log_dir,
            merged_log_file=merged_log_file,
            keep_gpu_logs=False,
        )


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    pdirs_ = PATHS.patient_dirs()[0:1]
    calc_scores_for_pdirs(
        pdirs_,
        serial_processing=True,
        available_gpus=[0, 1, 2, 3],
    )
