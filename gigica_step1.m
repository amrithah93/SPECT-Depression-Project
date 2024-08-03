%setup data for batch analyses: depression dataset
%first, copy healthy SPECT patients files into rest_max (depression patients), then setup directories
%then move JobSubmit.sh and gigica files into rest_max using mv -t command from home directory 
%then set up subjectlist.txt file for input directories - remember to put the full path!

outputDir1 = ['/data/qneuromark/Data/Depression/amenclinic_depression_SPECT/Data/Depression_ICA_Analyses']; %output file 

prefix = 'Depression_ICA_Output_Rest';
refFiles = {'/trdapps/linux-x86_64/matlab/toolboxes/GroupICAT/GroupICAT/icatb/icatb_templates/Neuromark_fMRI_1.0.nii'};
%calling the Neuromark 1.0 template

parallel_info.mode = 'serial';
parallel_info.num_workers = 31;

modalityType = 'sMRI';
dataSelectionMethod = 4;
numOfSess = 1;
dummy_scans = 0;
keyword_designMatrix = 'no';
input_design_matrices = {};
group_ica_type = 'spatial';
algoType = 10;
which_analysis = 1;
%icasso_opts.sel_mode = 'randinit';  
%icasso_opts.num_ica_runs = 10; 
%icasso_opts.min_cluster_size = 8; 
%icasso_opts.max_cluster_size = 10; 
preproc_type = 4;
%maskFile = '';
pcaType = 1;
pca_opts.stack_data = 'yes';
pca_opts.precision = 'double';
pca_opts.tolerance = 1e-4;
pca_opts.max_iter = 1000;
group_pca_type = 'subject specific';
backReconType = 5;
scaleType = 2;
numReductionSteps = 2;
doEstimation = 0; 
estimation_opts.PC1 = 'max';
estimation_opts.PC2 = 'mean';
numOfPC1 = 53;
numOfPC2 = 53;
perfType = 3;

%make subject txt file by copying folder IDs into one file - controls first than depression patients
input_data_file_patterns = cellstr(readlines(['/data/qneuromark/Data/Depression/amenclinic_depression_SPECT/Data/rest_max/depression_subject_list_final.txt']));

display_results.formatName = 'html'; 
display_results.convert_to_zscores = 'yes';
display_results.threshold = 1.0;
display_results.image_values = 'positive and negative';
display_results.slice_plane = 'axial';
display_results.network_summary_opts.comp_network_names = {'SC', (1:5); 'AU', (6:7); 'SM', (8:16); 'VI', (17:25); 'CC', (26:42); 'DM', (43:49); 'CB', (50:53)};
display_results.network_summary_opts.image_values = display_results.image_values;
display_results.network_summary_opts.threshold = display_results.threshold;
display_results.network_summary_opts.convert_to_z = display_results.convert_to_zscores;
