attach(depression_dataset) #modified dep dataset with first row removed to account for first missing subject in initial ICA analyses
library(corrplot)

#run correlation matrix to see which variables are related

install.packages("ggplot2")
install.packages("reshape2")
library(ggplot2)
library(reshape2)

#set up correlation
#omit missing data and create dataset for GSC items 1 and 2
#you have to use cbind else it won't work 
#repeat process for multiple data sets to see which variables correlate

# Define combinations of variables to analyze for each correlation pair
#GSC_A has 9 items, so most combos is 36 x 36
correlation_combo_list <-list(
  c("GSC_A_1_Feeling_depressed_or_being_in_a_sad_mood", "GSC_A_2_Having_a_decreased_interest_in_things_that_are_usually_fun_including_sex"),
  c("GSC_A_2_Having_a_decreased_interest_in_things_that_are_usually_fun_including_sex", "GSC_A_3_Experiencing_a_significant_change_in_weight_or_appetite_increased_or_decreased"),
  c("GSC_A_3_Experiencing_a_significant_change_in_weight_or_appetite_increased_or_decreased", "GSC_A_4_Having_recurrent_thoughts_of_death_or_suicide"),
  c("GSC_A_4_Having_recurrent_thoughts_of_death_or_suicide", "GSC_A_5_Experiencing_sleep_changes_such_as_a_lack_of_sleep_or_a_marked_increase_in_sleep"),
  c("GSC_A_5_Experiencing_sleep_changes_such_as_a_lack_of_sleep_or_a_marked_increase_in_sleep", "GSC_A_6_Feeling_physically_agitated_or_being_slowed_down"),
  c("GSC_A_6_Feeling_physically_agitated_or_being_slowed_down", "GSC_A_7_Having_feelings_of_low_energy_or_tiredness"),
  c("GSC_A_7_Having_feelings_of_low_energy_or_tiredness", "GSC_A_8_Having_feelings_of_worthlessness_helplessness_hopelessness_or_guilt"),
  c("GSC_A_8_Having_feelings_of_worthlessness_helplessness_hopelessness_or_guilt", "GSC_A_9_Experiencing_decreased_concentration_or_memory"))


#rename variables for readability
GSC1 <-GSC_A_1_Feeling_depressed_or_being_in_a_sad_mood
GSC2 <-GSC_A_2_Having_a_decreased_interest_in_things_that_are_usually_fun_including_sex
GSC3 <-GSC_A_3_Experiencing_a_significant_change_in_weight_or_appetite_increased_or_decreased
GSC4 <-GSC_A_4_Having_recurrent_thoughts_of_death_or_suicide
GSC5 <-GSC_A_5_Experiencing_sleep_changes_such_as_a_lack_of_sleep_or_a_marked_increase_in_sleep
GSC6 <-GSC_A_6_Feeling_physically_agitated_or_being_slowed_down
GSC7<-GSC_A_7_Having_feelings_of_low_energy_or_tiredness
GSC8<-GSC_A_8_Having_feelings_of_worthlessness_helplessness_hopelessness_or_guilt
GSC9 <-GSC_A_9_Experiencing_decreased_concentration_or_memory

#create data frame
df <-cbind.data.frame(GSC1, GSC2, GSC3, GSC4, GSC5, GSC6, GSC7, GSC8, GSC9)

# Create a data frame for each pair and omit NAs
cor1 <-na.omit(cbind(GSC1, GSC2))
cor2 <-na.omit(cbind(GSC2,GSC3))
cor3 <-na.omit(cbind(GSC3,GSC4))
cor4 <-na.omit(cbind(GSC4,GSC5))
cor5 <-na.omit(cbind(GSC5, GSC6))
cor6 <-na.omit(cbind(GSC6,GSC7))
cor7 <-na.omit(cbind(GSC7,GSC8))
cor8 <-na.omit(cbind(GSC8,GSC9))
full_dataset <-na.omit(cbind(df))
                
# Calculate the correlation coefficient
correlations_pair_1<- cor(cor1, method = c("pearson")) #remember to remove nas or it won't work
correlations_pair_2<- cor(cor2, method = c("pearson")) #remember to remove nas or it won't work
correlations_pair_3<- cor(cor3, method = c("pearson")) #remember to remove nas or it won't work
correlations_pair_4<- cor(cor4, method = c("pearson")) #remember to remove nas or it won't work
correlations_pair_5<- cor(cor5, method = c("pearson")) #remember to remove nas or it won't work
correlations_pair_6<- cor(cor6, method = c("pearson")) #remember to remove nas or it won't work
correlations_pair_7<- cor(cor7, method = c("pearson")) #remember to remove nas or it won't work
correlations_pair_8<- cor(cor8, method = c("pearson")) #remember to remove nas or it won't work
corr_full <-cor(full_dataset, method = c("pearson")) #full dataset

#correlation plots for all 8 pairs separately 
ggcorrplot(cor(cor1)) #correlation matrix 1; make sure to zoom into figure to see values
ggcorrplot(correlations_pair_1,
           hc.order = TRUE,
           type = "full",
           lab = TRUE)

ggcorrplot(cor(cor2)) #correlation matrix 2; make sure to zoom into figure to see values
ggcorrplot(correlations_pair_2,
           hc.order = TRUE,
           type = "full",
           lab = TRUE)

ggcorrplot(cor(cor3)) #correlation matrix 3; make sure to zoom into figure to see values
ggcorrplot(correlations_pair_2,
           hc.order = TRUE,
           type = "full",
           lab = TRUE)

ggcorrplot(cor(cor4)) #correlation matrix 4; make sure to zoom into figure to see values
ggcorrplot(correlations_pair_4,
           hc.order = TRUE,
           type = "full",
           lab = TRUE)

ggcorrplot(cor(cor5)) #correlation matrix 5; make sure to zoom into figure to see values
ggcorrplot(correlations_pair_5,
           hc.order = TRUE,
           type = "full",
           lab = TRUE)

ggcorrplot(cor(cor6)) #correlation matrix 6; make sure to zoom into figure to see values
ggcorrplot(correlations_pair_6,
           hc.order = TRUE,
           type = "full",
           lab = TRUE)


ggcorrplot(cor(cor7)) #correlation matrix 7; make sure to zoom into figure to see values
ggcorrplot(correlations_pair_7,
           hc.order = TRUE,
           type = "full",
           lab = TRUE)


ggcorrplot(cor(cor8)) #correlation matrix 8; make sure to zoom into1 figure to see values
ggcorrplot(correlations_pair_8,
           hc.order = TRUE,
           type = "full",
           lab = TRUE)

#correlation plot of all together
ggcorrplot(corr_full) #correlation matrix full
ggcorrplot(corr_full,
           hc.order = TRUE,
           type = "full",
           lab = TRUE)

#do some preliminary ML classification
install.packages("OneR")
