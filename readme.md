# Abbreviations 
Various abbreviations throughout the code:  
`szr`: seizure  
`ann`: annotation  
`mtz`: main time zone (of a patient)
`ptnt`: patient  
`seg`: segment  
`esegs`: existing segments
`dur`: duration  
`sig`: signal  
`chn`: channel  
`acfw`: autocorrelation function width  


# General Notes
* There are two EEG channels in the data: the distal (D) and proximal (P).  


# todo update with new dataset steps
# Pipeline Steps
1. cleaning_annotations
   1. clean_txt_annotations
   2. convert_txt_to_tabular
   3. combine_visit_annotations
   4. check_and_transfer_annotations
   5. localize_annotations
2. preprocessing
   1. validate_patients
   2. segment_tables
   3. train_test_split
3. feature_extraction
   1. extract_features