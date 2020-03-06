# NORMALIZATION SETTINGS

NORM_FUNCS = ['amplitude', 'zscore'] #options are zscore, amplitude, gradient, maxabs, norm. Including both should only work with concatenation method
NORM_OPTIONS = [{}, {'normalize':True}] #kwargs for the normalization functions above
NORM_METHOD = ['inplace', 'concatenate'] #options are inplace or concatenate. If concatenate, the model must be set up to take input vectors with greater than one feature.

# FILE NAMES
PRED_FNAME = 'predictions'
WEIGHT_FNAME = 'model_weights'

# ACTIVATION SETTINGS 

ALPHA = 0.3 # alpha for paramaterized relu or LeakyReLU

# PAD SETTINGS

PAD_MODE = 'constant'
PAD_CV = 0