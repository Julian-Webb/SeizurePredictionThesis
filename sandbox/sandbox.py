from pathlib import Path
from config.paths import PATHS
from model_eval.calc_segment_probabilities import calc_ptnt_seg_probabilities
from models.CNN import create_ptnt_cnn
from utils import QueuedCall
from models.FB_MLP import create_ptnt_ensemble_and_save
from utils import run_queued_calls_on_gpus
import numpy as np
from sklearn.metrics import precision_recall_curve


def check_files_contained_in_both():
    src = Path('/data/home/webb/seizure_annotations/STEP1_original_anns/toadd')
    dst = Path('/data/home/webb/seizure_annotations/STEP1_original_anns/')

    for src_pdir in sorted(list(src.iterdir())):
        dst_pdir = dst / src_pdir.name
        dst_files = [f.name for f in dst_pdir.iterdir()]
        for file in src_pdir.iterdir():
            if file.name in dst_files:
                # print(f'✓ File contained: {file.parent.name}/{file.name}')
                pass
            else:
                print(f'x File not in dst: {file.parent.name}/{file.name}')


def metrics():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_pred_proba = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])

    precision_recall_curve(y_true, y_pred_proba)


def train_models():
    tasks = []
    for pdir in PATHS.patient_dirs():
        tasks.append(QueuedCall(func=create_ptnt_cnn, args=(pdir,), label=f"{pdir.name} - CNN"))
        tasks.append(QueuedCall(func=create_ptnt_ensemble_and_save, args=(pdir,), label=f"{pdir.name} - FB-MLP"))

    run_queued_calls_on_gpus(tasks=tasks, gpus=[1, 2, 3])


def predictions():
    pdirs = PATHS.patient_dirs()[1:]

    tasks = []
    for pdir in pdirs:
        tasks.append(
            QueuedCall(func=calc_ptnt_seg_probabilities, args=(pdir,), label=f'{pdir.name} - calc_seg_probabilities'))


    run_queued_calls_on_gpus(tasks=tasks, gpus=[1, 2, 3])


if __name__ == '__main__':
    pass
    # predictions()
