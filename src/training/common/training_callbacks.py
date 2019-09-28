import json
from datetime import datetime
import numpy as np
from tensorflow import keras as K


def default(val):
    if isinstance(val, np.float32):
        return float(val)
    raise TypeError


class ModelAndLogSavingCallback(K.callbacks.Callback):
    def __init__(self, model, output_path):
        super().__init__()
        self.output_path = output_path != ''
        self.logs = []
        self.model = model

    def on_epoch_end(self, epoch, logs={}):
        t = datetime.now().isoformat().split('.')[0]
        K.models.save_model(self.model, self.output_path + f'md{t}.h5')

        with open(self.output_path + 'log.json', 'w+') as fo:
            json.dump(self.logs, fo, default=default, indent=4)

        self.logs.append(logs)
        return

    def on_train_end(self, logs={}):
        return
