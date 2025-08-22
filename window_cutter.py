import os
import pyedflib
import datetime
import numpy as np


class WindowCutter:
    def __init__(self):
        pass


if __name__ == '__main__':
    data_dir = r'/Users/julian/Developer/EEG Data/'
    # data_folder = r'/data/datasets/'
    data_path = os.path.join(data_dir, r'20240201_UNEEG_ForMayo/B52K3P3G/V5a')

    file_path = os.path.join(data_path, 'B52K3P3G_01_0004245_20211008_01_EEGdata.edf')
    edf = pyedflib.EdfReader(file_path)

    start_date_time = datetime.datetime(edf.startdate_year, edf.startdate_month, edf.startdate_day, edf.starttime_hour, edf.starttime_minute, edf.starttime_second, edf.starttime_subsecond)
    duration = datetime.timedelta(seconds=edf.file_duration)
    end_date_time = start_date_time + duration

    print(f'{start_date_time=}')
    print(f'{duration=}')
    print(f'{end_date_time=}')

    n_signals = edf.signals_in_file
    signals = np.zeros((n_signals, edf.getNSamples()[0]))
    for i in range(n_signals):
        signals[i, :] = edf.readSignal(i)


    print()
    # go through the folders where the data is stored
    # for edf_file in os.listdir(data_path):
    #     # pyedflib.EdfReader()
