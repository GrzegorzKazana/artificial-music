import json
import os
from datetime import datetime
import numpy as np
from tensorflow import keras as K
import matplotlib.pyplot as plt


def default(val):
    if isinstance(val, np.float32):
        return float(val)
    raise TypeError


class ModelAndLogSavingCallback(K.callbacks.Callback):
    def __init__(self, model, output_path):
        super().__init__()
        self.output_path = output_path
        self.logs = []
        self.model = model

    def on_epoch_end(self, epoch, logs={}):
        t = datetime.now().isoformat().split('.')[0]
        K.models.save_model(self.model, self.output_path + f'md{t}.h5')

        with open(os.path.join(self.output_path, 'log.json'), 'w+') as fo:
            json.dump(self.logs, fo, default=default, indent=4)

        self.logs.append(logs)
        return


class GeneratingAndPlottingCallback(K.callbacks.Callback):
    def __init__(self, model, sample_generator, seed_generator, output_path=''):
        super().__init__()
        self.output_path = output_path
        self.model = model
        self.seed_generator = seed_generator
        self.sample_generator = sample_generator

    def on_epoch_end(self, epoch, logs={}):
        t = datetime.now().isoformat().split('.')[0]
        seed = self.seed_generator()
        sample = self.sample_generator(self.model, seed)

        _, axs = plt.subplots(nrows=4, ncols=4, figsize=(30, 10),
                              subplot_kw={'xticks': [], 'yticks': []})
        for ax, x_ in zip(axs.flat, sample):
            ax.imshow(x_.T[::-1, :])
        plt.tight_layout()
        res = plt.gcf()
        plt.show()

        if self.output_path != '':
            res.save_fig(os.path.join(self.output_path, f'{t}.png'))

        return
