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
   4. list_rename_move_edf_data
   5. remove_P4Hk23M7L_files_from_2000
2. preprocessing
   1. estimate_seizure_starts
   2. validate_patients
   3. segment_tables
   4. train_test_split
3. feature_extraction
   1. extract_features