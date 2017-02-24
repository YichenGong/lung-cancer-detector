'''
Base class for the model

Contains the boiler-plate code for a model in tensorflow
plus a decorator class inspired by: https://gist.github.com/danijar/8663d3bbfd586bffecf6a0094cd116f2

Boiler plate code includes:
1. Creating summaries for all variables/outputs to be visualized
in tensorboard

2. Create Computation Graph -> Optimize -> Calculate Error
'''