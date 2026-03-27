# Abbreviations
Various abbreviations throughout the code:  
`szr`: seizure  
`ann`: annotation  
`mtz`: main time zone (of a patient)  
`ptnt`: patient  
`pdir`: patient directory 
`seg`: segment  
`esegs`: existing segments  
`dur`: duration  
`sig`: signal  
`chn`: channel  
`acfw`: autocorrelation function width

# General Notes
* There are two EEG channels in the data: the distal (D) and proximal (P).

# Pipeline Steps
1. cleaning_annotations
    1. clean_txt_annotations
    2. convert_txt_to_tabular
    3. combine_visit_annotations
    4. check_and_transfer_annotations
    5. localize_annotations
2. preprocessing
    1. validate_patients
    2. filter_signals
    3. segment_tables
    4. train_test_allocation
    5. feature_extraction.extract_features
3. models.train_models
4. model_eval
   1. calc_segment_probabilities
   2. clips
   3. event_based_metrics
   4. eval_models
5. model_comparison.comparision_table
6. cycle_extraction
