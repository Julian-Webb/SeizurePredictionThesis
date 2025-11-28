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
   1. file_correction
   2. annotations_to_csv
   3. combine_annotations
   4. rename_and_move_edf_data
2. preprocessing
   1. estimate_seizure_starts
   2. validate_patients
   3. segment_tables
   4. train_test_split
   5. extract_features