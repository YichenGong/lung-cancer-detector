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
#import tensorflow as tf

import options
opt = options.parse()

import importlib
import random

#Set the random seed for further usage
random.seed(opt.seed)
#tf.set_random_seed(opt.seed)

#Import the data loader
dl = (importlib.import_module("dataloader." + opt.data)).get_data_loader(opt)

# for x, y in dl.get_next():
# 	print(x.shape, len(y))

# dl.validate()
# for x, y in dl.get_next():
# 	print(x.shape, len(y))

# dl.test()
# for x, y in dl.get_next():
# 	print(x.shape, len(y))

#Load the model
#model = (importlib.import_module("models." + opt.model)).get_model()