# Abbreviations 
Various abbreviations throughout the code:  
`szr`: seizure  
`ann`: annotation  
`ptnt`: patient  

# Pipeline Steps
1. data_cleaning
   1. file_correction.py
   2. annotations_to_csv.py
   3. rename_and_move_edf_data.py
2. preprocessing
   1. combine_annotations.py
   2. estimate_seizure_starts.py
   3. valid_participants
   4. window_cutter.py