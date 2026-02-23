from pathlib import Path

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



if __name__ == '__main__':
    metrics()