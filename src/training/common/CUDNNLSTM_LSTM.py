from tensorflow import keras as K
import json
from copy import deepcopy
import os


def cudnnlstm_to_lstm(cuda_model):
    """
    model still needs to be compiled for further training
    """
    cuda_json = json.loads(cuda_model.to_json())
    cpu_json = deepcopy(cuda_json)

    cpu_json['config']['layers'] = [transformCUDNNLSTMlayer(
        layer) if layer['class_name'] == 'CuDNNLSTM' else layer for layer in cpu_json['config']['layers']]

    cpu_model = K.models.model_from_json(json.dumps(cpu_json))

    cuda_model.save_weights('w_transfer.h5')
    cpu_model.load_weights('w_transfer.h5')

    os.remove('w_transfer.h5')

    return cpu_model


def transformCUDNNLSTMlayer(cudnnlstm_layer):
    lstm_layer = deepcopy(cudnnlstm_layer)
    lstm_layer['class_name'] = 'LSTM'
    lstm_layer['config']['activation'] = 'tanh'
    lstm_layer['config']['recurrent_activation'] = 'sigmoid'
    lstm_layer['config']['dropout'] = 0
    lstm_layer['config']['recurrent_dropout'] = 0
    lstm_layer['config']['implementation'] = 1
    lstm_layer['config']['unroll'] = False
    lstm_layer['config']['use_bias'] = True

    return lstm_layer
