import numpy as np
import json
from data_processing import load_data, pad_traces, _normalize
from model_generator import model_generator
from upeak_settings import NORM_FUNCS, NORM_OPTIONS, NORM_METHOD
from upeak_settings import PAD_MODE, PAD_CV

def predict_peaks(traces, model=None, weights=None, normalize=False):
    
    orig_length = traces.shape[1]
    traces, model = _parse_inputs_to_expected(traces, model, weights, normalize)

    result = model.predict(traces)
    return result[:, :orig_length, :]
    
def _parse_inputs_to_expected(traces, model, weights, normalize):
    # model should be option_dict loaded from json file - parse path to loaded dictionary
    # traces should be without nans and padded if needed (though padding is not possible without knowing model size...)

    if model is None:
        # set default model parameters
        model_dict = {'dims': (64, 1), 'classes': 3, 'steps': 3, 'layers': 2, 'filters': 32, 'kernel_size': 8, 
            'stride': 1, 'transfer': True, 'activation': 'relu', 'padding': 'same'}
    elif type(model) == str:
        with open(model, 'r') as json_file:
            model_dict = json.load(json_file)
    elif type(model) == dict:
        model_dict = model
    else:
        raise TypeError('Model data type not understood')

    traces = load_data(traces)
    if normalize:
        traces = _normalize(NORM_FUNCS, NORM_OPTIONS, NORM_METHOD, traces)
    model_depth = model_dict['steps']
    traces = pad_traces(traces, model_depth, pad_mode=PAD_MODE, cv=PAD_CV)

    input_dims = (traces.shape[1], traces.shape[2], model_dict['classes'])
    model = model_generator(input_dims=input_dims, steps=model_dict['steps'], conv_layers=model_dict['layers'],
        filters=model_dict['filters'], kernel_size=model_dict['kernel'], strides=model_dict['stride'],
        transfer=model_dict['transfer'], activation=model_dict['activation'], padding=model_dict['padding'])

    if weights is None:
        #this should load some sort of default weight that goes with the default model
        pass
    else:
        model.load_weights(weights)
    
    return traces, model