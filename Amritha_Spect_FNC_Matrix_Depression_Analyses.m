%% set up main variables for analyses  
%first check the order of the files and how they were run
sesInfo.inputFiles.name % lists the output of the 'files' file in 'SPECT_outputSubject.mat' to check if healthy vs. patients were run first

a = spm_read_vols(spm_vol('Depression_ICA_Output_Rest_group_loading_coeff_.nii'));

size(a); %tells you the dimensions of a, in this case it's a subject ...
         % by component matrix of 213 x 53

%you can save it as a txt file as follows:
save("loadings.txt","a","-ascii")
%save transposed dep and healthy loading parameters
%first depression loadings
save("dep_loadings","depind_transp","-ascii")
%now healthy loadings
save("healthy_loadings","hcind_transp","-ascii")

%save relevant clinical variables to export to R for linear regression analysis - depression variables
save("depression_age","age","-ascii")
save("depression_sex","sex_coded","-ascii")

%% test a preliminary GLM model
%convert variables to table to run regression
%convert DepLoadingParameters as a string to run
%convert dataset into an array

DepLoadingParameters_String = num2str(DepLoadingParameters); %convert to run GLM
X = [age, sex_coded, GSC_A_49_Having_a_marked_decreased_interest_in_important_activi]; 
y = (DepLoadingParameters); 

%test on first parameter
y1 = y(:,1); %first parameter
mdl_test = fitlm(X,y1); %test model and check output

%test on last parameter
y53 = y(:,53); %last parameter
mdl_test_last = fitlm(X,y53); %test model and check output

%% now, construct a for loop to do the GLM iteratively across all 23627 x 53 matrix
%matrix then a for loop for glm

% Initialize a cell array to store the models
models = cell(1, size(DepLoadingParameters, 2));

% Loop through each response variable and fit a GLM
for i = 1:size(DepLoadingParameters, 2)
    y = DepLoadingParameters(:,i);
    mdl = fitglm(X,y);
    models{i} = mdl; % Store the model in the cell array
    
    % Display or further process the model
    disp(['Model for DepLoadingParameter ', num2str(i), ':']);
    disp(mdl);
end

 % Extract model summary
    summaryStr = evalc('disp(mdl)');
    modelSummaries{i} = summaryStr;
    
 %transpose models variable output for easier viewing
 models = models.'; 
 
 %% now, display coefficients especially beta and p values
 %% first get the coefficients  
% Assuming 'models' is a cell array containing the GLM models
% models{1,1}, models{2,1}, ..., models{53,1} where 53 is the number of models

% Determine the number of models
numModels = numel(models);
 
% Determine the number of predictors (including intercept)
numPredictors = size(models{1,1}.Coefficients, 1);

% Initialize matrix to store p-values
pValues = zeros(numModels, numPredictors);

% Loop through each model and extract coefficients
for i = 1:numModels
    % Extract the coefficients table from the model
     coeffTable = models{i,1}.Coefficients; 
     
    % Store the coefficients table in the cell array
    coefficients{i} = coeffTable;
    coefficients = coefficients.'; %transpose for readability 
end

%% check output for sanity check 
    
% Display the coefficients for the first model as an example and sanity
% check
disp('Coefficients for the first model:');
disp(coefficients{1});

%% Loop through each model and extract p values
for i = 1:numModels
    % Extract the p-values from the coefficients table
    p_values = models{i,1}.Coefficients.pValue;
    
    % Store the p-values in the matrix
    pValues(i, :) = p_values';
    
    %rename variable to keep it separate from t-test p
    regression_pValues = pValues; 
end


%% now get beta values
% Determine the number of predictors (including intercept)
numPredictors = size(models{1,1}.Coefficients, 1);

% Preallocate a matrix to store the beta coefficients
betaCoefficients = zeros(numModels, numPredictors);

% Loop through each model and extract beta coefficients
for i = 1:numModels
    % Extract the beta coefficients (estimates) from the coefficients table
    betas = models{i,1}.Coefficients.Estimate;
    
    % Store the beta coefficients in the matrix
    betaCoefficients(i, :) = betas';
    
    %rename variables to keep it separate from t-test betas
    regression_betas = betaCoefficients; 
end

%% get t values

% Determine the number of predictors (including intercept)
numPredictors = size(models{1,1}.Coefficients, 1);

% Preallocate a matrix to store the beta coefficients
tValues = zeros(numModels, numPredictors);

% Assuming 'models' is a cell array containing the GLM models
% models{1,1}, models{2,1}, ..., models{53,1} where 53 is the number of models

% Determine the number of models
numModels = 53;


% Loop through each model and extract beta coefficients
for i = 1:numModels
    % Extract the t values (estimates) from the coefficients table
    tValuesCurrent = coefficients{i,1}.tStat; % t-values for this model
    % Store the t-values and transpose
    tValues(i, :) = tValuesCurrent';
end


%%  Rows correspond to different models, and columns correspond to predictors including the intercept
  
 %double check p values for signage and significant p values for intercept,
 %age, sex, and GSC_A_49 

 %first set up variables for plotting, specifically for each of the
 %variable beta coefficients
regression_betas_intercept = regression_betas(:,1); %intercept
regression_betas_age = regression_betas(:,2); %age
regression_betas_sex = regression_betas(:,3); %sex
regression_betas_GSC_A_49 = regression_betas(:,4); %GSC_A_49

%next, do the same for p values
regression_pValues_intercept = regression_pValues(:,1); %intercept
regression_pValues_age = regression_pValues(:,2); %age
regression_pValues_sex = regression_pValues(:,3); %sex
regression_pValues_GSC_A_49 = regression_pValues(:,4); %GSC_A_49

%now do plots for each of the four variables for betas and p values and
%plot it as imagesc to see values more clearly 


%% check histograms of data to see distribution before determining color axis
histogram(regression_pValues_intercept); %intercept p values
title('Intercept Regression P Values')
histogram(regression_pValues_age); %age p values
title('Age Regression P Values')
histogram(regression_pValues_sex); %sex p values
title('Sex Regression P Values')
histogram(regression_pValues_GSC_A_49); %GSC 49 p values
title('GSC Item 49 Regression P Values')

%now do same for beta values

histogram(regression_betas_intercept); %intercept beta values
title('Intercept Regression Beta Values')
histogram(regression_betas_age); %age beta values
title('Age Regression Beta Values')
histogram(regression_betas_sex); %sex beta values
title('Sex Regression Beta Values')
histogram(regression_betas_GSC_A_49); %GSC 49 beta values
title('GSC 49 Regression Beta Values')

%% intercept
regression_intercept_plot = (-log10(regression_pValues_intercept).*sign(regression_betas_intercept)); %intercept 
max_value = max(abs(regression_intercept_plot(:)));
imagesc(regression_intercept_plot); %display as an image to check p values
title('Regression Intercept Log Values'); 
% Add a color bar
colorbar;
% Adjust the color scale to the data's range
caxis([-max_value max_value]); % Set the color limits from 0 to the maximum value
% Set colormap (optional)
colormap('jet'); % Choose a colormap like 'jet', 'hot', 'cool', etc.


%% age
regression_age_plot = (-log10(regression_pValues_age).*sign(regression_betas_age)); %age  
max_value = max(abs(regression_age_plot(:)));
imagesc(regression_age_plot); %display as an image to check p values
title('Regression Age Log Values'); 
% Add a color bar
colorbar;
% Adjust the color scale to the data's range
caxis([-max_value max_value]); % Set the color limits from 0 to the maximum value
% Set colormap (optional)
colormap('jet'); % Choose a colormap like 'jet', 'hot', 'cool', etc.


%% sex
regression_sex_plot = (-log10(regression_pValues_sex).*sign(regression_betas_sex)); %sex  
max_value = max(abs(regression_sex_plot(:)));
imagesc(regression_sex_plot); %display as an image to check p values
title('Regression Sex Log Values'); 
% Add a color bar
colorbar;

% Adjust the color scale to the data's range
caxis([-max_value max_value]); % Set the color limits from 0 to the maximum value default
% Set colormap (optional)
colormap('jet'); % Choose a colormap like 'jet', 'hot', 'cool', etc.

%% GSC_A_49
regression_GSC_A_49_plot = (-log10(regression_pValues_GSC_A_49).*sign(regression_betas_GSC_A_49)); %GSC_A_49  
disp(max(regression_GSC_A_49_plot)); %determine max value which is 8.2704
max_value = max(abs(regression_GSC_A_49_plot(:)));
imagesc(regression_GSC_A_49_plot); %display as an image to check p values
title('Regression GSC Item 49 (Depression Coded Item) Log Values'); 
% Add a color bar
colorbar;

% Adjust the color scale to the data's range
caxis([-max_value max_value]); % Set the color limits from 0 to the maximum value

% Set colormap (optional)
colormap('jet'); % Choose a colormap like 'jet', 'hot', 'cool', etc.

%% display all in one image
plot_all = -log10(pValues).*sign(betaCoefficients); %plot all values in one image
save('plot_all.mat','plot_all'); 
max_value = max(abs(plot_all(:)));
imagesc(plot_all); %plot all values
title('All Values: -log10(pValues).*sign(betaCoefficients) from 53 x 4 subject coefficient matrix'); 
% Add a color bar
colorbar;

% Adjust the color scale to the data's range
caxis([-max_value max_value]); % Set the color limits from min to the maximum value

% Set colormap (optional)
colormap('jet'); % colormap settings

%% plot just t values
imagesc(tValues); 
max_value = max(abs(tValues(:)));
% Add a color bar
colorbar;

% Adjust the color scale to the data's range
caxis([-max_value max_value]); % Set the color limits from min to the maximum value

% Set colormap (optional)
colormap('jet'); % colormap settings

%set title
title('T Values for All Four Predictors'); 

 %% set up indices for depression patients and healthy controls 
 
 depind = 1:23627; %dep subjects
 hcind = 23628:23703; %healthy controls

 %compute mean of hc and sz for the 53 components
 hcmn = mean(a(hcind,:));  
 save("hcmean.txt", "hcmn", "-ascii")
 depmn = mean(a(depind,:));  
 save("depmean.txt", "depmn", "-ascii")
 %save as txt files for R analysis

 %% create variables to see relationships between FNC and clinical variables
 %transpose indices so that plots can be rendered, rename variables

 hcind_transp = hcind'; %transpose hc indices variable
 depind_transp = depind'; %transpose tb indices variable
 hcmn_transp = hcmn'; %transpose hc mean variable
 depmn_transp = depmn'; %transpose tb mean variable

 %two sample t-test to compare HC vs dep
 [h,p,ci,stats]=ttest2(a(depind,:),a(hcind,:)); 
 save("pvals_dep_uncorrected.txt","p","-ascii") % save p values into text file
 
 qvals_ttest_corrected = mafdr(p, 'BHFDR', true);  %fdr correction - obtain q values for t.test results
 save("qvals_dep_corrected.txt","qvals_ttest_corrected","-ascii") % save p values into text file

  
 %take a look at the component with the largest t-value
 %the other way to compare is to look at the structural network
 %connectivity or SNC, the cross-correlation among ICA loading parameters.
 
%bar plot with loading parameter values for figure
bar(depmn); %plot average FNC for 53 components in patients
title('Depression Mean Loading Components')
xlabel('Loading Components Number (1-53) from Neuromark')
ylabel('Loading Component Value')

bar(hcmn); %plot average FNC for 53 components for controls
title('Control Mean Loading Components')
xlabel('Loading Components Number (1-53) from Neuromark')
ylabel('Loading Component Value')

%group differences plot
B = bar(depmn-hcmn); %group differences between hc - sz means
title('Patients-Controls Differences Plotted: Mean Loading Components')
xlabel('Loading Components Number (1-53) from Neuromark')
ylabel('Loading Component Value')

%display bar plots with different colors for domains 
% Define the data and highlight significant qvals with FDR corrected
% components
data = stats.tstat; % data dep-healthy
 
% Define positions for each bar for clarity in plotting
positions = 1:length(data);
 
% Plot the bars individually to set specific colors
hold on; % Allows multiple bars to be plotted on the same figure
bar1 = bar(positions(1:5), data(1:5), 'FaceColor', 'green'); % SC Network
bar2 = bar(positions(6:7), data(6:7), 'FaceColor', 'magenta'); %AUD Network
bar3 = bar(positions(8:16), data(8:16), 'FaceColor', 'blue'); %SM Network
bar4 = bar(positions(17:25), data(17:25), 'FaceColor', 'red'); %VIS Network
bar5 = bar(positions(26:42), data(26:42), 'FaceColor', 'yellow'); %CC Network
bar6 = bar(positions(43:49), data(43:49), 'FaceColor', 'cyan'); %DM Network
bar7 = bar(positions(50:53), data(50:53), 'FaceColor', 'black'); %CB Network
title('Depression-Healthy Differences Plotted: Mean Loading Components')
xlabel('Loading Components Number (1-53) from Neuromark')
ylabel('Loading Component Value')
x = 1:53; %label components for the tickmarks
xticks(x);  %set the tickmarks
L = {'SC Network', 'AUD Network', 'SM Network', 'VIS Network', 'CC Network', 'DM Network', 'CB Network'}; %define legend
lgd=legend(L,'Location','southwest');
hold off;

%double check signage and plot t values to see how they look
plot(-log10(p).*sign(stats.tstat)); %plot log values of p and t values

%now do a two sample t-test to compare HC vs SZ means and use it to plot
%sig results
[h,p,ci,stats]=ttest2(a(hcmn,:),a(szmn,:));
%% get spatial maps for 15 components for HC > SZ results
%

    files = {'SPECT_output_group_component_ica_.nii,4',... 
    'SPECT_output_group_component_ica_.nii,6',...
    'SPECT_output_group_component_ica_.nii,16',... 
    'SPECT_output_group_component_ica_.nii,18',... 
    'SPECT_output_group_component_ica_.nii,19',...
    'SPECT_output_group_component_ica_.nii,23',...
    'SPECT_output_group_component_ica_.nii,29',...
    'SPECT_output_group_component_ica_.nii,30',...
    'SPECT_output_group_component_ica_.nii,31',...
    'SPECT_output_group_component_ica_.nii,32',...
    'SPECT_output_group_component_ica_.nii,34',...
    'SPECT_output_group_component_ica_.nii,38',...
    'SPECT_output_group_component_ica_.nii,40',...
    'SPECT_output_group_component_ica_.nii,41',...
    'SPECT_output_group_component_ica_.nii,47'};

icatb_image_viewer(files,'display_type','montage','structfile',fullfile(fileparts(which('gift.m')),'icatb_templates','ch2bet.nii'), ...
    'threshold', 3.0, 'slices_in_mm', (-60:8:60), 'convert_to_zscores', 'yes', 'image_values', 'positive','iscomposite','yes');

%% display FNC plots 

  %display scz FNC only
  display_FNC(corr(a(tbind,:)),.5)
  title ('TBI FNC map')

  %display controls FNC only
  display_FNC(corr(a(hcind,:)),.5)
  title ('controls FNC map')

  %plot FNC matrix tb - healthy
  display_FNC(corr(a(tbind,:))-corr(a(hcind,:)));
  title ('53x53 FNC plot of tbi-healthy differences uncorrected')
  
  %display just the lower half of triangle uncorrected results
  display_FNC(tril(corr(a(tbind,:))-corr(a(hcind,:))));
  title('Lower Half of 53x53 FNC plot uncorrected')
  tri_lower = tril(corr(a(tbind,:))-corr(a(hcind,:)));

  %display upper half of the triangle with FDR corrected differences
  display_FNC(triu((C1)));
  title('Upper Half of 53x53 FNC plot FDR corrected')
  tri_upper = triu(C1); 

  %display both upper and lower triangles in a new figure
  tri_full = tri_upper + tri_lower; 
  display_FNC(tri_full,.5);
  title('Uncorrected and corrected matrices tbi-healthy') 

  %corrected results only
  display_FNC(q); 
  title('corrected matrix only') 

  %% do FDR correction and plot subsequent q vals

  d = (corr(a(tbind,:))-corr(a(hcind,:))); %first plot the tbi-hc differences
 
  %now get correlations of full loading param matrix and transform to vector
  N = 1824; %number of subjects
  fnc = icatb_mat2vec(d); %loading parameters for all 
  pvals = 2*(1-tcdf(r_to_t(satlin(abs(fnc)),N-1),N-1)); %get pvalues  
  save("pvals_uncorrected.txt","pvals","-ascii")
  
  %do fdr and bonferonni correction for group differences in resting state
  %data
  qvals_differences_rest = mafdr(pvals, 'BHFDR', true);  %fdr correction - obtain q values
  save("qvals_corrected.txt","qvals_differences_rest","-ascii")
  qvals_differences_rest = icatb_vec2mat(qvals_differences_rest);
  q = qvals_differences_rest; %rename for connectogram feature

%% now make variables and q vals for patients only
   
  fnc_patients = Patients_Loadings; 
  N = 1748; %number of subjects for patients
  fnc_patients = vec_patients; %rename vector to fnc variable
  pvals_patients = 2*(1-tcdf(r_to_t(satlin(abs(fnc_patients)),N-1),N-1)); %get pvalues
  save("pvals_uncorrected_patients.txt","pvals_patients","-ascii")
  
  %do fdr and bonferonni correction to compare results
  qvals_patients = mafdr(pvals_patients, 'BHFDR', true);  %fdr correction - obtain q values
  save("qvals_corrected_patients.txt","qvals_patients","-ascii")

  vec_patients_matrix = icatb_vec2mat(vec_patients); %retransform matrix just to have on hand
  save("vec_patients_matrix.txt","vec_patients_matrix","-ascii")
 
%% do connectogram for fdr corrected p vals for healthy-patients which survived FDR correction
%do for all 53 components, and threshold accordingly 
%Nifti file name
fname = 'TBI_ICA_Output_Rest_group_component_ica_.nii'; %your component maps

C1 = d.*(icatb_vec2mat(qvals_differences_rest<=0.05)); %this zeros out correlations that are greater than 0.05
C = C1; %rename

%Network names and components relative to the component nifti file entered above in variable fname
comp_network_names = {'SC Network', [1,2,3,4,5]; % SC Network
   'AUD Network', [6,7];   % AUD network
   'SM Network', [8,9,10,11,12,13,14,15,16]; % Sensorimotor comps
   'VIS Network', [17,18,19,20,21,22,23,24,25]; % Visual comps
   'CC Network', [26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42]; % CC Network
   'DMN Network', [43,44,45,46,47,48,49]
    'CB Network', [50,51,52,53]};  % DMN Network    
     
%Slice view
%threshold picked after comparing q values to correlations corresponding to a q = 0.05

icatb_plot_connectogram([], comp_network_names, 'C', C, 'threshold', 2 , 'conn_threshold', 0.32, 'image_file_names', fname, 'colorbar_label', 'Corr', 'display_type', 'render', 'convert_to_zscores',0, 'slice_plane', 'sagittal', 'exclude_zeros', 1, 'CLIM', [-0.5,0.5]);

%% do connectogram for healthy-patients for all nodes  
%do for all 53 components, and threshold accordingly 
%Nifti file name
fname = 'SPECT_output_group_component_ica_.nii'; %your component maps

C2 = corr(Healthy_Loadings)-corr(Patients_Loadings); %hc-sz differences 
 %hc-sz differences 
C = C2; %rename

%Network names and components relative to the component nifti file entered above in variable fname
comp_network_names = {'SC Network', [1,2,3,4,5]; % SC Network
   'AUD Network', [6,7];   % AUD network
   'SM Network', [8,9,10,11,12,13,14,15,16]; % Sensorimotor comps
   'VIS Network', [17,18,19,20,21,22,23,24,25]; % Visual comps
   'CC Network', [26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42]; % CC Network
   'DMN Network', [43,44,45,46,47,48,49]
    'CB Network', [50,51,52,53]}; % DMN Network    
     
%Slice view
%threshold picked after comparing q values to correlations corresponding to a q = 0.05

icatb_plot_connectogram([], comp_network_names, 'C', C, 'threshold', 2 , 'conn_threshold', 0.3, 'image_file_names', fname, 'colorbar_label', 'Corr', 'display_type', 'render', 'convert_to_zscores',0, 'slice_plane', 'sagittal', 'exclude_zeros', 1);
 
%% connectogram for patient regressions with 11 components 
fname = 'SPECT_output_group_component_ica_.nii'; %your component maps

%Network names and components relative to the component nifti file entered above in variable fname
S = corr(szind,:); %new FDR corrected matrix for patients
C = S.*(icatb_vec2mat(qvals_patients<=0.05)); %this zeros out Css that are greater than 0.05

%removed associations with sex and age as different constructs to show;
%only showing hearing voices
comp_network_names = {
   'CC Network', [31]; % CC Network
  };
 
%Slice view

icatb_plot_connectogram([], comp_network_names, 'C', C, 'threshold', 2, 'conn_threshold', 0.7 , 'image_file_names', fname, 'colorbar_label', 'Corr', 'display_type', 'render', 'convert_to_zscores', 'no', 'slice_plane', 'sagittal', 'exclude_zeros', 0);


%Slice view

icatb_plot_connectogram([], comp_network_names, 'C', C, 'threshold', 2, 'conn_threshold', 0.3 , 'image_file_names', fname, 'colorbar_label', 'Corr', 'display_type', 'render', 'convert_to_zscores', 'no', 'slice_plane', 'sagittal', 'exclude_zeros', 0);


%% run regressions with corrected hc-sz matrices%%%%

sz_dataset = schiz83123dxclassessymptoms; %rename schizophrenia dataset as new variable
sz_dataset_vector = rows2vars(szdataset); %make into a vector
modelspec = 'q ~ age + sex_coded + GSC_A_98_Hearing_voices_or_sounds_that_are_not_real'; 
mdl1 = fitglm(sz_dataset, modelspec, 'linear', 'Distribution', 'normal');


%% GLM for patients
Patients_Loadings = a(77:end,:);
Predictors = [sz_age, sz_sex_coded, sz_GSC_A_98_Hearing_voices_or_sounds_that_are_not_real];

% 137 patients
for idx = 1:size(Patients_Loadings,2)

    regression_results = fitlm(Patients_Loadings(:,idx), Predictors);
    Results{idx} = regression_results;
    
end

%% GLM for controls

Control_Loadings = a(1:76:end,:);
Predictors_Controls = [hc_age, hc_sex_coded, hc_GSC_A_98_Hearing_voices_or_sounds_that_are_not_real];
% 76 controls
for idx = 1:size(Control_Loadings,2)
    regression_results_controls = fitglm(Predictors_Controls); Control_Loadings(:,idx);
    Results{idx} = regression_results_controls; %ignore error message
end

%% save results
save("Patients_Loadings.txt","Patients_Loadings","-ascii")

%% do a quick sanity check on bar plots

modelspec = 'Patients_Loadings ~ sz_age + sz_sex_coded + GSC_A_98_Hearing_voices_or_sounds_that_are_not_real + GSC_A_98_Hearing_voices_or_sounds_that_are_not_real';
mdl_bar_plot = fitglm(modelspec,'Distribution','binomial'); 

%% GLM for sex differences and diagnosis

modelspec = 'Patients_Loadings ~ sz_sex_coded';
mdl_patients_sex = fitglm(Patients_Loadings,modelspec,'Distribution','binomial');

%% now get q vals for regression results for age, sex, GSC Item 98 and 99
qvals_differences_regression_model8_age = mafdr(pval_age, 'BHFDR', true);  %fdr correction - obtain q values
save("qvals_age_hallucinations_model.txt","qvals_differences_regression_model8_age","-ascii")
qvals_differences_regression_model8_sex = mafdr(pval_sex, 'BHFDR', true);  %fdr correction - obtain q values
save("qvals_sex_hallucinations_model.txt","qvals_differences_regression_model8_sex","-ascii")
qvals_differences_regression_model8_item98 = mafdr(pval_item98, 'BHFDR', true);  %fdr correction - obtain q values
save("qvals_item98_hallucinations_model.txt","qvals_differences_regression_model8_item98","-ascii")
qvals_differences_regression_model8_item99 = mafdr(pval_item99, 'BHFDR', true);  %fdr correction - obtain q values
save("qvals_item99_hallucinations_model.txt","qvals_differences_regression_model8_item99","-ascii")
%obtain q vals for dx regression
qvals_differences_regression__dx = mafdr(pvals_dx_regression, 'BHFDR', true);  %fdr correction - obtain q values
%take p values from the R run regression with age, sex, and diagnosis, and correct for FDR to get
%results