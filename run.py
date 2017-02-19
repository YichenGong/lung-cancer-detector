'''
Contains the main template of the experiment

1. Get options

2. Load the appropriate dataloader 
(in the correct mode -> training/testing/validation etc. dpending upon the options)

3. Load the appropriate model (depending upon the options)

4. Run the model in the mode specified

This one script takes care of all the boilerplate code
needed to run an experiment on the dataset and subsequently 
create a Kaggle Submission file (if the mode includes testing)
'''
import options

opt = options.parse()
print(opt)