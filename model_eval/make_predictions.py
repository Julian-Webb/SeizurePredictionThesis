import pandas as pd
from keras import Model

from config.paths import PatientDir
from utils.io import pickle_path

pdir: PatientDir = PatientDir('/data/home/webb/UNEEG/datasets/ultra2/U002-DE01-07')
segs = pd.read_pickle(pickle_path(pdir.segments_table))
esegs = segs[segs['exists']].drop(columns=['exists'])


# todo unlike for training, we neither want to shuffle nor subsample segs here.