from collections import OrderedDict

import pandas as pd

# The sampling frequency of all EEG signals in the edf files
SAMPLING_FREQUENCY_HZ = 207.0310546581987

# The EEG channels in their *universal order*. D: distal, P: proximal
CHANNELS = ['EEG SQ_D-SQ_C', 'EEG SQ_P-SQ_C']
N_CHANNELS = len(CHANNELS)

# The shift used to go from a single marker to the estimated start
# (as calculated in estimate_seizure_starts)
single_marker_to_start_shift: pd.Timedelta | None = None

# The minimum number of valid seizures for a patient to be included in the analysis
MIN_VALID_SEIZURES_PER_PATIENT = 10

# How much of a patient's total recording timespan must be recorded for him/her to be valid
MIN_RATIO_RECORDED_TO_BE_VALID = 0.4

# How much of the data's entire timespan should be allocated to training the models
RATIO_OF_TIMESPAN_FOR_TRAINING = 0.6

# How the EEG bands are defined (lower frequency, upper frequency, name)
# Note: dicts maintain insertion order in Python
# todo is the lower delta and upper gamma bound correct?
SPECTRAL_BANDS = {'Delta': (0.5, 4), 'Theta': (4, 8), 'Alpha': (8, 12), 'Beta': (12, 35), 'Gamma': (35, 100)}

# In the data, there are much more interictal that preictal segments (about 100 to 1)
# During training, we want to subsample the interictal segments to reach a ratio that works well for training.
MAX_INTERICTAL_TO_PREICTAL_SEGMENT_RATIO = 20