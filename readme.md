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

# General Notes
* There are two EEG channels in the data: the distal (D) and proximal (P).  



# Pipeline Steps
1. data_cleaning
   1. file_correction
   2. convert_txt_annotations
   3. combine_annotations
   4. rename_and_move_edf_data
2. preprocessing
   1. timezone_adjustment
   2. estimate_seizure_starts
   3. validate_patients
   4. segment_tables
   5. train_test_split
   6. extract_features