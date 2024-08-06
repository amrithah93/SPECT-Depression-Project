%% A. set up main variables for analyses  
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

%% B. test a preliminary GLM model with age, sex, GSC 49 in it. 
%convert variables to table to run regression
%convert DepLoadingParameters as a string to run
%convert dataset into an array
X = [age, sex_coded, GSC_A_49_Having_a_marked_decreased_interest_in_important_activi]; 
y = (DepLoadingParameters); 

y1 = y(:,1); %first parameter
mdl_test = fitlm(X,y1); %test model and check output

%test on last parameter
y53 = y(:,53); %last parameter
mdl_test_last = fitlm(X,y53); %test model and check output

%% C. Run the first GLM test with age, sex, and GSC item 49
% Initialize a cell array to store the models
mdl_test = cell(1, size(DepLoadingParameters, 2));
currentResponse = DepLoadingParameters(:,i);

% Loop through each response variable and fit a GLM
for i = 1:size(DepLoadingParameters, 2)
    y = DepLoadingParameters(:,i);
    mdl= fitglm(X,y);
    models{i} = mdl; % Store the model in the cell array
    
    % Display or further process the model
    disp(['Multivariate Model Regressed Against DepLoadingParameter ', num2str(i), ':']);
    disp(mdl);
    % Extract model summary
    summaryStr = evalc('disp(mdl)');
    modelSummaries{i} = summaryStr;
end

models = models'; %transpose
 
 %% D. now, display coefficients especially beta and p values
 %% E. first get the beta coefficients  

% Extract and store the beta coefficients
    betaCoefficients(i, :) = mdl.Coefficients.Estimate';
    
%% F. Get the p values next

% Loop through each model and extract beta coefficients
for i = 1:numModels
    % Extract the t values (estimates) from the coefficients table
    pValMatrix(i, :) = models{i,1}.Coefficients.pValue';
    regression_pValues = pValMatrix; %rename for ease
end
 
%% G. get t values

% Loop through each model and extract beta coefficients
for i = 1:numModels
    % Extract the t values (estimates) from the coefficients table
    tValuesCurrent = coefficients{i,1}.tStat; % t-values for this model
    % Store the t-values and transpose
    tValues(i, :) = tValuesCurrent';
end

%%  H. Rows correspond to different models, and columns correspond to predictors including the intercept
  
 %double check p values for signage and significant p values for intercept,
 %age, sex, and GSC_A_49 

 %first set up variables for plotting, specifically for each of the
 %variable beta coefficients
regression_betas_intercept = betaCoefficients(:,1); %intercept
regression_betas_age = betaCoefficients(:,2); %age
regression_betas_sex = betaCoefficients(:,3); %sex
regression_betas_GSC_A_49 = betaCoefficients(:,4); %GSC_A_49

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
plot_all = -log10(regression_pValues).*sign(betaCoefficients); %plot all values in one image
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

%% now do GLM for multiple GSC variables as a multiple regression
%%omit any missing values from following variables if needed
age_cleaned = rmmissing(age);  % Remove age rows with missing values
sex_coded_cleaned = rmmissing(sex_coded); %remove missing sex
GSC_A_1_cleaned = rmmissing(GSC_A_1_Feeling_depressed_or_being_in_a_sad_mood); %16565 size double
GSC_A_2_cleaned = rmmissing(GSC_A_2_Having_a_decreased_interest_in_things_that_are_usually_); %16434 size double
GSC_A_3_cleaned = rmmissing(GSC_A_3_Experiencing_a_significant_change_in_weight_or_appetite); %16208 size double
GSC_A_4_cleaned = rmmissing(GSC_A_4_Having_recurrent_thoughts_of_death_or_suicide); %16289 size double
GSC_A_5_cleaned = rmmissing(GSC_A_5_Experiencing_sleep_changes_such_as_a_lack_of_sleep_or_a); %16375 size double
GSC_A_6_cleaned = rmmissing(GSC_A_6_Feeling_physically_agitated_or_being_slowed_down); %16326 size double
GSC_A_7_cleaned = rmmissing(GSC_A_7_Having_feelings_of_low_energy_or_tiredness); %16501 size double
GSC_A_8_cleaned = rmmissing(GSC_A_8_Having_feelings_of_worthlessness_helplessness_hopelessn); %16503 size double
GSC_A_9_cleaned = rmmissing(GSC_A_9_Experiencing_decreased_concentration_or_memory); %16486 size double

%Run all GSC variables in one model
Xall = [age, sex_coded, GSC_A_1_Feeling_depressed_or_being_in_a_sad_mood, GSC_A_2_Having_a_decreased_interest_in_things_that_are_usually_...,
    GSC_A_3_Experiencing_a_significant_change_in_weight_or_appetite, GSC_A_4_Having_recurrent_thoughts_of_death_or_suicide...,
    GSC_A_5_Experiencing_sleep_changes_such_as_a_lack_of_sleep_or_a, GSC_A_6_Feeling_physically_agitated_or_being_slowed_down...,
    GSC_A_7_Having_feelings_of_low_energy_or_tiredness, GSC_A_8_Having_feelings_of_worthlessness_helplessness_hopelessn...,
    GSC_A_9_Experiencing_decreased_concentration_or_memory]; %12 predictors including intercept
y = (DepLoadingParameters); 

% Initialize a cell array to store the models
models_multivariate = cell(1, size(DepLoadingParameters, 2));

% Loop through each response variable and fit a GLM
for i = 1:size(DepLoadingParameters, 2)
    y = DepLoadingParameters(:,i);
    mdl_multivariate = fitglm(Xall,y);
    models_multivariate{i} = mdl_multivariate; % Store the model in the cell array
    
    % Display or further process the model
    disp(['Multivariate Model Regressed Against DepLoadingParameter ', num2str(i), ':']);
    disp(mdl_multivariate);
    % Extract model summary
    summaryStr_multivariate = evalc('disp(mdl_multivariate)');
    multivariate_modelSummaries{i} = summaryStr_multivariate;
end
    
 %transpose models variable output for easier viewing
 models_multivariate = models_multivariate'; 
 
% Now, loop through each model and extract p values
% Number of models and predictors
numModels = 53;   
numPredictors = size(models_multivariate{1}.Coefficients.pValue, 1); %12

% Initialize a matrix to store p-values for each model
pValues_Multivariate = zeros(numModels, numPredictors); %53 x 12 

% Loop through each model
for i = 1:numModels
    % Extract the p-values from the coefficients table
    pValues = models_multivariate{i, 1}.Coefficients.pValue; %make sure variable is pValues to extract table info
    
    % Store the p-values in the matrix
    pValues_Multivariate(i, :) = pValues'; %NOW make sure variable is what you defined matrix to be
end

% Now get beta coefficients for easier viewing
  
% Number of models and predictors
numModels = 53;  
numPredictors = size(models_multivariate{1}.Coefficients.Estimate, 1); %12

% Initialize a matrix to store coefficients for each model
betaCoefficients_multivariate = zeros(numModels, numPredictors); %53 x 12

% Loop through each model
for i = 1:numModels
    % Extract the beta coefficients from the coefficients table
    betaCoefficients = models_multivariate{i, 1}.Coefficients.Estimate;
    
    % Store the beta coefficients in the matrix (transpose to row vector)
    betaCoefficients_multivariate(i, :) = betaCoefficients';
end
%% log plot multivariate model

%now run log graphs for each model (plot all) for the 9 GSC variables and
%age, sex, and the intercept

regression_betas_intercept_all = betaCoefficients_multivariate(:,1); % intercept
regression_betas_age_all = betaCoefficients_multivariate(:,2); % age
regression_betas_sex_all = betaCoefficients_multivariate(:,3); % sex
regression_betas_GSC1_all = betaCoefficients_multivariate(:,4); % GSC Item 1
regression_betas_GSC2_all = betaCoefficients_multivariate(:,5); % GSC Item 2
regression_betas_GSC3_all = betaCoefficients_multivariate(:,6); % GSC Item 3
regression_betas_GSC4_all = betaCoefficients_multivariate(:,7); % GSC Item 4
regression_betas_GSC5_all = betaCoefficients_multivariate(:,8); % GSC Item 5
regression_betas_GSC6_all = betaCoefficients_multivariate(:,9); % GSC Item 6
regression_betas_GSC7_all = betaCoefficients_multivariate(:,10); % GSC Item 7
regression_betas_GSC8_all = betaCoefficients_multivariate(:,11); % GSC Item 8
regression_betas_GSC9_all = betaCoefficients_multivariate(:,12); % GSC Item 9

regression_pValues_intercept_all = pValues_Multivariate(:,1); % intercept
regression_pValues_age_all = pValues_Multivariate(:,2); % age
regression_pValues_sex_all = pValues_Multivariate(:,3); % sex
regression_pValues_GSC1_all = pValues_Multivariate(:,4); % GSC Item 1
regression_pValues_GSC2_all = pValues_Multivariate(:,5); % GSC Item 2
regression_pValues_GSC3_all = pValues_Multivariate(:,6); % GSC Item 3
regression_pValues_GSC4_all = pValues_Multivariate(:,7); % GSC Item 4
regression_pValues_GSC5_all = pValues_Multivariate(:,8); % GSC Item 5
regression_pValues_GSC6_all = pValues_Multivariate(:,9); % GSC Item 6
regression_pValues_GSC7_all = pValues_Multivariate(:,10); % GSC Item 7
regression_pValues_GSC8_all = pValues_Multivariate(:,11); % GSC Item 8
regression_pValues_GSC9_all = pValues_Multivariate(:,12); % GSC Item 9

%determinin min pvals and plot
minValue_intercept_all = min(regression_pValues_intercept_all(:));
minValue_age_all = min(regression_pValues_age_all(:));
minValue_sex_all = min(regression_pValues_sex_all(:));
minValue_GSC1_all = min(regression_pValues_GSC1_all(:));
minValue_GSC2_all = min(regression_pValues_GSC2_all(:));
minValue_GSC3_all = min(regression_pValues_GSC3_all(:));
minValue_GSC4_all = min(regression_pValues_GSC4_all(:));
minValue_GSC5_all = min(regression_pValues_GSC5_all(:));
minValue_GSC6_all = min(regression_pValues_GSC6_all(:));
minValue_GSC7_all = min(regression_pValues_GSC7_all(:));
minValue_GSC8_all = min(regression_pValues_GSC8_all(:));
minValue_GSC9_all = min(regression_pValues_GSC9_all(:));

disp(['The minimum p-value for the intercept across all models is: ', num2str(minValue_intercept_all)]);
disp(['The minimum p-value for the age across all models is: ', num2str(minValue_age_all)]);
disp(['The minimum p-value for the sex across all models is: ', num2str(minValue_sex_all)]);
disp(['The minimum p-value for the GSC1 across all models is: ', num2str(minValue_GSC1_all)]);
disp(['The minimum p-value for the GSC2 across all models is: ', num2str(minValue_GSC2_all)]);
disp(['The minimum p-value for the GSC3 across all models is: ', num2str(minValue_GSC3_all)]);
disp(['The minimum p-value for the GSC4 across all models is: ', num2str(minValue_GSC4_all)]);
disp(['The minimum p-value for the GSC5 across all models is: ', num2str(minValue_GSC5_all)]);
disp(['The minimum p-value for the GSC6 across all models is: ', num2str(minValue_GSC6_all)]);
disp(['The minimum p-value for the GSC7 across all models is: ', num2str(minValue_GSC7_all)]);
disp(['The minimum p-value for the GSC8 across all models is: ', num2str(minValue_GSC8_all)]);
disp(['The minimum p-value for the GSC9 across all models is: ', num2str(minValue_GSC9_all)]);

% Concatenate the minimum values into one variable
minValues_all = [
    minValue_intercept_all;
    minValue_age_all;
    minValue_sex_all;
    minValue_GSC1_all;
    minValue_GSC2_all;
    minValue_GSC3_all;
    minValue_GSC4_all;
    minValue_GSC5_all;
    minValue_GSC6_all;
    minValue_GSC7_all;
    minValue_GSC8_all;
    minValue_GSC9_all
];

all_beta_values = [
    regression_betas_intercept_all;  
    %regression_betas_age_all;  %not needed for new display
    %regression_betas_sex_all;  %not needed for new display
    regression_betas_GSC1_all;
    regression_betas_GSC2_all;
    regression_betas_GSC3_all;
    regression_betas_GSC4_all;
    regression_betas_GSC5_all;
    regression_betas_GSC6_all;
    regression_betas_GSC7_all;
    regression_betas_GSC8_all;
    regression_betas_GSC9_all
];

all_p_values = [
    regression_pValues_intercept_all;
    %regression_pValues_age_all; %not needed for new display
    %regression_pValues_sex_all; %not needed for new display
    regression_pValues_GSC1_all;
    regression_pValues_GSC2_all;
    regression_pValues_GSC3_all;
    regression_pValues_GSC4_all;
    regression_pValues_GSC5_all;
    regression_pValues_GSC6_all;
    regression_pValues_GSC7_all;
    regression_pValues_GSC8_all;
    regression_pValues_GSC9_all
];

%% display multivariate models in one image regarding log plots

 % Compute the plot data as a log function
 plot_model = -log10(all_p_values) .* sign(all_beta_values);
 plot(plot_model); 
 title('Log Function of PVals and BetaVals for 9 GSC Variables'); 
 saveas(gcf, sprintf('Multivariate_GSC1-9.png', model_num));
 
 %plot as an image
 imagesc(plot_model);
 title('Image of PVals and BetaVals for 9 GSC Variables'); 

%% model 9 models separately as effects are weak and see what results are
%set up variables
%now run 9 separate GLMs for each

% Create a cell array to store the independent variables for each model
% order

X1 = [age, sex_coded, GSC_A_1_Feeling_depressed_or_being_in_a_sad_mood];
X2 = [age, sex_coded, GSC_A_2_Having_a_decreased_interest_in_things_that_are_usually_fun];
X3 = [age, sex_coded, GSC_A_3_Experiencing_a_significant_change_in_weight_or_appetite];  
X4 = [age, sex_coded, GSC_A_4_Having_recurrent_thoughts_of_death_or_suicide];
X5 = [age, sex_coded, GSC_A_5_Experiencing_sleep_changes_such_as_a_lack_of_sleep_or_a]; 
X6 = [age, sex_coded, GSC_A_6_Feeling_physically_agitated_or_being_slowed_down];  
X7 = [age, sex_coded, GSC_A_7_Having_feelings_of_low_energy_or_tiredness];
X8 = [age, sex_coded, GSC_A_8_Having_feelings_of_worthlessness_helplessness_hopelessn]; 
X9 = [age, sex_coded, GSC_A_9_Experiencing_decreased_concentration_or_memory];

% Loop through each response variable and fit a GLM

%Start with Model 1
for i = 1:size(DepLoadingParameters, 2)
    y = DepLoadingParameters(:,i);
    mdl_1= fitglm(X1, y);
    model_1{i} = mdl_1; % Store the model in the cell array
    % Display or further process the model
    model_1 = model_1.'; %transpose
end

% Model 2

for i = 1:size(DepLoadingParameters, 2)
    y = DepLoadingParameters(:,i);
    mdl_2= fitglm(X2, y);
    model_2{i} = mdl_2; % Store the model in the cell array
   % Display or further process the model
    model_2 = model_2.'; %transpose
end

% Model 3

for i = 1:size(DepLoadingParameters, 2)
    y = DepLoadingParameters(:,i);
    mdl_3= fitglm(X3, y);
    model_3{i} = mdl_3; % Store the model in the cell array
   % Display or further process the model
    model_3 = model_3.'; %transpose
end

% Model 4

for i = 1:size(DepLoadingParameters, 2)
    y = DepLoadingParameters(:,i);
    mdl_4= fitglm(X4, y);
    model_4{i} = mdl_4; % Store the model in the cell array
   % Display or further process the model
    model_4 = model_4.'; %transpose
end

% Model 5

for i = 1:size(DepLoadingParameters, 2)
    y = DepLoadingParameters(:,i);
    mdl_5= fitglm(X5, y);
    model_5{i} = mdl_5; % Store the model in the cell array
   % Display or further process the model
    model_5 = model_5.'; %transpose
end

% Model 6

for i = 1:size(DepLoadingParameters, 2)
    y = DepLoadingParameters(:,i);
    mdl_6= fitglm(X6, y);
    model_6{i} = mdl_6; % Store the model in the cell array
   % Display or further process the model
    model_6 = model_6.'; %transpose
end

% Model 7

for i = 1:size(DepLoadingParameters, 2)
    y = DepLoadingParameters(:,i);
    mdl_7= fitglm(X7, y);
    model_7{i} = mdl_7; % Store the model in the cell array
   % Display or further process the model
    model_7 = model_7.'; %transpose
end

% Model 8

for i = 1:size(DepLoadingParameters, 2)
    y = DepLoadingParameters(:,i);
    mdl_8= fitglm(X8, y);
    model_8{i} = mdl_8; % Store the model in the cell array
   % Display or further process the model
    model_8 = model_8.'; %transpose
end

% Model 9

for i = 1:size(DepLoadingParameters, 2)
    y = DepLoadingParameters(:,i);
    mdl_9= fitglm(X9, y);
    model_9{i} = mdl_9; % Store the model in the cell array
   % Display or further process the model
    model_9 = model_9.'; %transpose
end

%% Loop through each model and extract p values

%Model 1
% Loop through each model and extract p values
for i = 1:numModels
    % Extract the p values from the coefficients table
    pValMatrix1(i, :) = model_1{i,1}.Coefficients.pValue';
    regression_pValues = pValMatrix1; %rename for ease
end

%Model 2

for i = 1:numModels
    % Extract the p values from the coefficients table
    pValMatrix2(i, :) = model_2{i,1}.Coefficients.pValue';
    regression_pValues = pValMatrix2; %rename for ease
end

%Model 3

for i = 1:numModels
    % Extract the p values from the coefficients table
    pValMatrix3(i, :) = model_3{i,1}.Coefficients.pValue';
    regression_pValues = pValMatrix3; %rename for ease
end

%Model 4

for i = 1:numModels
    % Extract the p values from the coefficients table
    pValMatrix4(i, :) = model_4{i,1}.Coefficients.pValue';
    regression_pValues = pValMatrix4; %rename for ease
end

%Model 5

for i = 1:numModels
    % Extract the p values from the coefficients table
    pValMatrix5(i, :) = model_5{i,1}.Coefficients.pValue';
    regression_pValues = pValMatrix5; %rename for ease
end

%Model 6

for i = 1:numModels
    % Extract the p values from the coefficients table
    pValMatrix6(i, :) = model_6{i,1}.Coefficients.pValue';
    regression_pValues = pValMatrix6; %rename for ease
end


%Model 7

for i = 1:numModels
    % Extract the p values from the coefficients table
    pValMatrix7(i, :) = model_7{i,1}.Coefficients.pValue';
    regression_pValues = pValMatrix7; %rename for ease
end


%Model 8

for i = 1:numModels
    % Extract the p values from the coefficients table
    pValMatrix8(i, :) = model_8{i,1}.Coefficients.pValue';
    regression_pValues = pValMatrix8; %rename for ease
end

%Model 9

for i = 1:numModels
    % Extract the p values from the coefficients table
    pValMatrix9(i, :) = model_9{i,1}.Coefficients.pValue';
    regression_pValues = pValMatrix9; %rename for ease
end

%% plot log function of each of these models (Models 1-9)

 %double check p values for signage and significant p values for intercept,
 %age, sex, and GSC Items 1-9
 
%% first, extract beta coefficients from the 9 models
% Initialize a cell array to store the models
% Number of models and predictors
numModels = 53;
numPredictors = 4;

% Initialize matrices to store coefficients for each model set
betaCoefficients1 = zeros(numModels, numPredictors);
betaCoefficients2 = zeros(numModels, numPredictors);
betaCoefficients3 = zeros(numModels, numPredictors);
betaCoefficients4 = zeros(numModels, numPredictors);
betaCoefficients5 = zeros(numModels, numPredictors);
betaCoefficients6 = zeros(numModels, numPredictors);
betaCoefficients7 = zeros(numModels, numPredictors);
betaCoefficients8 = zeros(numModels, numPredictors);
betaCoefficients9 = zeros(numModels, numPredictors);

% Loop through each model
for i = 1:numModels
    % Extract coefficients for each model set
    betaCoefficients1(i, :) = model_1{i, 1}.Coefficients.Estimate(1:numPredictors)';
    betaCoefficients2(i, :) = model_2{i, 1}.Coefficients.Estimate(1:numPredictors)';
    betaCoefficients3(i, :) = model_3{i, 1}.Coefficients.Estimate(1:numPredictors)';
    betaCoefficients4(i, :) = model_4{i, 1}.Coefficients.Estimate(1:numPredictors)';
    betaCoefficients5(i, :) = model_5{i, 1}.Coefficients.Estimate(1:numPredictors)';
    betaCoefficients6(i, :) = model_6{i, 1}.Coefficients.Estimate(1:numPredictors)';
    betaCoefficients7(i, :) = model_7{i, 1}.Coefficients.Estimate(1:numPredictors)';
    betaCoefficients8(i, :) = model_8{i, 1}.Coefficients.Estimate(1:numPredictors)';
    betaCoefficients9(i, :) = model_9{i, 1}.Coefficients.Estimate(1:numPredictors)';
end

%% then, get the p values

% Number of models and predictors
numModels = 53;
numPredictors = 4;

% Initialize matrices to store p-values for each model set
pValues1 = zeros(numModels, numPredictors);
pValues2 = zeros(numModels, numPredictors);
pValues3 = zeros(numModels, numPredictors);
pValues4 = zeros(numModels, numPredictors);
pValues5 = zeros(numModels, numPredictors);
pValues6 = zeros(numModels, numPredictors);
pValues7 = zeros(numModels, numPredictors);
pValues8 = zeros(numModels, numPredictors);
pValues9 = zeros(numModels, numPredictors);

% Loop through each model
for i = 1:numModels
    % Extract p-values for each model set
    pValues1(i, :) = model_1{i, 1}.Coefficients.pValue(1:numPredictors)';
    pValues2(i, :) = model_2{i, 1}.Coefficients.pValue(1:numPredictors)';
    pValues3(i, :) = model_3{i, 1}.Coefficients.pValue(1:numPredictors)';
    pValues4(i, :) = model_4{i, 1}.Coefficients.pValue(1:numPredictors)';
    pValues5(i, :) = model_5{i, 1}.Coefficients.pValue(1:numPredictors)';
    pValues6(i, :) = model_6{i, 1}.Coefficients.pValue(1:numPredictors)';
    pValues7(i, :) = model_7{i, 1}.Coefficients.pValue(1:numPredictors)';
    pValues8(i, :) = model_8{i, 1}.Coefficients.pValue(1:numPredictors)';
    pValues9(i, :) = model_9{i, 1}.Coefficients.pValue(1:numPredictors)';
end

 %Model 1
 %first set up variables for plotting, specifically for each of the
 %variable beta coefficients and respective pValues

% Model 1 coefficients and p-values (as already defined)
regression_betas_intercept_1 = betaCoefficients1(:,1); % intercept
regression_betas_age_1 = betaCoefficients1(:,2); % age
regression_betas_sex_1 = betaCoefficients1(:,3); % sex
regression_betas_GSC1 = betaCoefficients1(:,4); % GSC Item 1

% Model 1 p-values
regression_pValues_intercept_1 = pValues1(:,1); % intercept
regression_pValues_age_1 = pValues1(:,2); % age
regression_pValues_sex_1 = pValues1(:,3); % sex
regression_pValues_GSC1_1 = pValues1(:,4); % GSC Item 

%Model 2

regression_betas_intercept_2 = betaCoefficients2(:,1); % intercept
regression_betas_age_2 = betaCoefficients2(:,2); % age
regression_betas_sex_2 = betaCoefficients2(:,3); % sex
regression_betas_GSC2 = betaCoefficients2(:,4); % GSC Item 2

regression_pValues_intercept_2 = pValues2(:,1); % intercept
regression_pValues_age_2 = pValues2(:,2); % age
regression_pValues_sex_2 = pValues2(:,3); % sex
regression_pValues_GSC2 = pValues2(:,4); % GSC Item 2

%Model 3

regression_betas_intercept_3 = betaCoefficients3(:,1); % intercept
regression_betas_age_3 = betaCoefficients3(:,2); % age
regression_betas_sex_3 = betaCoefficients3(:,3); % sex
regression_betas_GSC3 = betaCoefficients3(:,4); % GSC Item 3

regression_pValues_intercept_3 = pValues3(:,1); % intercept
regression_pValues_age_3 = pValues3(:,2); % age
regression_pValues_sex_3 = pValues3(:,3); % sex
regression_pValues_GSC3 = pValues3(:,4); % GSC Item 3

%Model 4

regression_betas_intercept_4 = betaCoefficients4(:,1); % intercept
regression_betas_age_4 = betaCoefficients4(:,2); % age
regression_betas_sex_4 = betaCoefficients4(:,3); % sex
regression_betas_GSC4 = betaCoefficients4(:,4); % GSC Item 4

regression_pValues_intercept_4 = pValues4(:,1); % intercept
regression_pValues_age_4 = pValues4(:,2); % age
regression_pValues_sex_4 = pValues4(:,3); % sex
regression_pValues_GSC4 = pValues4(:,4); % GSC Item 4

%Model 5

regression_betas_intercept_5 = betaCoefficients5(:,1); % intercept
regression_betas_age_5 = betaCoefficients5(:,2); % age
regression_betas_sex_5 = betaCoefficients5(:,3); % sex
regression_betas_GSC5 = betaCoefficients5(:,4); % GSC Item 5

regression_pValues_intercept_5 = pValues5(:,1); % intercept
regression_pValues_age_5 = pValues5(:,2); % age
regression_pValues_sex_5 = pValues5(:,3); % sex
regression_pValues_GSC5 = pValues5(:,4); % GSC Item 5

%Model 6

regression_betas_intercept_6 = betaCoefficients6(:,1); % intercept
regression_betas_age_6 = betaCoefficients6(:,2); % age
regression_betas_sex_6 = betaCoefficients6(:,3); % sex
regression_betas_GSC6 = betaCoefficients6(:,4); % GSC Item 6

regression_pValues_intercept_6 = pValues6(:,1); % intercept
regression_pValues_age_6 = pValues6(:,2); % age
regression_pValues_sex_6 = pValues6(:,3); % sex
regression_pValues_GSC6 = pValues6(:,4); % GSC Item 6

%Model 7

regression_betas_intercept_7 = betaCoefficients7(:,1); % intercept
regression_betas_age_7 = betaCoefficients7(:,2); % age
regression_betas_sex_7 = betaCoefficients7(:,3); % sex
regression_betas_GSC7 = betaCoefficients7(:,4); % GSC Item 7

regression_pValues_intercept_7 = pValues7(:,1); % intercept
regression_pValues_age_7 = pValues7(:,2); % age
regression_pValues_sex_7 = pValues7(:,3); % sex
regression_pValues_GSC7 = pValues7(:,4); % GSC Item 7

%Model 8

regression_betas_intercept_8 = betaCoefficients8(:,1); % intercept
regression_betas_age_8 = betaCoefficients8(:,2); % age
regression_betas_sex_8 = betaCoefficients8(:,3); % sex
regression_betas_GSC8 = betaCoefficients8(:,4); % GSC Item 8

regression_pValues_intercept_8 = pValues8(:,1); % intercept
regression_pValues_age_8 = pValues8(:,2); % age
regression_pValues_sex_8 = pValues8(:,3); % sex
regression_pValues_GSC8 = pValues8(:,4); % GSC Item 8

%Model 9

regression_betas_intercept_9 = betaCoefficients9(:,1); % intercept
regression_betas_age_9 = betaCoefficients9(:,2); % age
regression_betas_sex_9 = betaCoefficients9(:,3); % sex
regression_betas_GSC9 = betaCoefficients9(:,4); % GSC Item 9

regression_pValues_intercept_9 = pValues9(:,1); % intercept
regression_pValues_age_9 = pValues9(:,2); % age
regression_pValues_sex_9 = pValues9(:,3); % sex
regression_pValues_GSC9 = pValues9(:,4); % GSC Item 9

%% determine which of the p values are lowest/most significant and from which
%of the 9 models 

% Number of models
numModels = 53;

% Initialize cell arrays to store the results for each model set
lowestPValues = cell(9, 1);
lowestPValueIndices = cell(9, 1);

% Loop through each model set from 1 to 9
for modelSet = 1:9
    % Construct the variable name for the current model set
    variableName = sprintf('pValues%d', modelSet);
    
    % Retrieve the p-values matrix for the current model set
    pValues_Multivariate = eval(variableName);
    
    % Initialize arrays to store the lowest p-value and its column index for each model
    lowestPValues{modelSet} = zeros(numModels, 1);
    lowestPValueIndices{modelSet} = zeros(numModels, 1);
    
    % Loop through each model
    for i = 1:numModels
        % Find the minimum p-value and its index in the current model (row)
        [lowestPValues{modelSet}(i), lowestPValueIndices{modelSet}(i)] = min(pValues_Multivariate(i, :));
    end
    
    % Display the results for the current model set
    disp(['Results for Model Set ', num2str(modelSet), ':']);
    for i = 1:numModels
        disp(['Model ', num2str(i), ':']);
        disp(['The lowest p-value is: ', num2str(lowestPValues{modelSet}(i))]);
        disp(['At column: ', num2str(lowestPValueIndices{modelSet}(i))]);
        disp(' ');
    end
end

%% display all in one image regarding log plots
% Process and plot for each model from 1 to 9
for model_num = 1:9
    % Dynamically generate variable names for p-values and coefficients
    pValuesVar = eval(sprintf('pValues%d', model_num));
    betaCoefficientsVar = eval(sprintf('betaCoefficients%d', model_num));

    % Compute the plot data
    plot_model = -log10(pValuesVar) .* sign(betaCoefficientsVar);

    % Save the plot data
    save(sprintf('plot_model%d.mat', model_num), 'plot_model');

    % Get the maximum value for color scaling
    max_value = max(abs(plot_model(:)));

    % Plot all values
    imagesc(plot_model);
    title(sprintf('All Coefficients from Model %d: Age, Sex, GSC%d', model_num, model_num)); 

    % Add a color bar
    colorbar;

    % Adjust the color scale to the data's range
    caxis([-max_value max_value]); % Set the color limits from min to the maximum value

    % Set colormap (optional)
    colormap('jet'); % colormap settings

    % Save the figure
    saveas(gcf, sprintf('Model_%d_Log_Plot.png', model_num));

    % Pause to allow the plot to be rendered (optional, useful if running interactively)
    pause(1);
end

%% now just plot p values for each of the 9 models

% Process and plot for each model from 1 to 9
for model_num = 1:9
    % Dynamically generate variable names for p-values
    pValuesVar = eval(sprintf('pValues%d', model_num));

    % Compute the -log10 of p-values
    plot_pValues = pValuesVar;

    % Save the plot data
    save(sprintf('plot_pValues_model%d.mat', model_num), 'plot_pValues');

    % Get the maximum value for color scaling
    max_value = max(abs(plot_pValues(:)));

    % Plot all p-values
    imagesc(plot_pValues);
    title(sprintf('P-values from Model %d: Age, Sex, GSC%d', model_num, model_num)); 

    % Add a color bar
    colorbar;

    % Adjust the color scale to the data's range
    caxis([0 max_value]); % Set the color limits from 0 to the maximum value

    % Set colormap (optional)
    colormap('jet'); % colormap settings

    % Save the figure
    saveas(gcf, sprintf('Model_%d_PValues_Plot.png', model_num));

    % Pause to allow the plot to be rendered (optional, useful if running interactively)
    pause(1);
end
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
