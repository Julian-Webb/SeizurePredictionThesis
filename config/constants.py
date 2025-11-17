import pandas as pd

# The sampling frequency of all EEG signals in the edf files
SAMPLING_FREQUENCY_HZ = 207.0310546581987

# The EEG channels
CHANNELS = ['EEG SQ_D-SQ_C', 'EEG SQ_P-SQ_C']
N_CHANNELS = len(CHANNELS)

# The shift used to go from a single marker to the estimated start
# (as calculated in estimate_seizure_starts)
single_marker_to_start_shift: pd.Timedelta | None = None

# The minimum number of valid seizures for a patient to be included in the analysis
MIN_VALID_SEIZURES_PER_PATIENT = 10
