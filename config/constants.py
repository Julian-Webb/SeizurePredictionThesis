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

# How much of a patient's total recording timespan must be recorded for him/her to be valid
MIN_RATIO_RECORDED_TO_BE_VALID = 0.4

# How much of the data's entire timespan should be allocated to training the models
RATIO_OF_TIMESPAN_FOR_TRAINING = 0.6


# How the EEG bands are defined (lower frequency, upper frequency, name)
# todo is the lower delta and upper gamma bound correct?
SPECTRAL_BANDS = [(0.5, 4, 'Delta'), (4, 8, 'Theta'), (8, 12, 'Alpha'), (12, 35, 'Beta'), (35, 100, 'Gamma')]

