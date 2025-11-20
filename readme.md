# Abbreviations 
Various abbreviations throughout the code:  
`szr`: seizure  
`ann`: annotation  
`ptnt`: patient  
`seg`: segment  
`dur`: duration  
`sig`: signal  
`chn`: channel  
`acfw`: autocorrelation function width  


# Pipeline Steps
1. data_cleaning
   1. file_correction.py
   2. annotations_to_csv.py
   3. combine_annotations.py
   4.rename_and_move_edf_data.py
2. preprocessing
   1. estimate_seizure_starts.py
   2. valid_participants
   3. window_cutter.py