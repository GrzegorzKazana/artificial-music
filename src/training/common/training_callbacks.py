import json
import os
from datetime import datetime
import numpy as np
from tensorflow import keras as K
import matplotlib.pyplot as plt
from gensim.models.callbacks import CallbackAny2Vec


def default(val):
    if isinstance(val, np.float32):
        return float(val)
    raise TypeError


class ModelAndLogSavingCallback(K.callbacks.Callback):
    def __init__(self, model, output_path='', save_log_only=False):
        super().__init__()
        self.output_path = output_path
        self.logs = []
        self.model = model
        self.save_log_only = save_log_only

        if output_path != '':
            model_json = model.to_json()
            with open(os.path.join(self.output_path, 'model.json'), 'w+') as fo:
                json.dump(json.loads(model_json), fo,
                          default=default, indent=4)

            if os.path.isfile(os.path.join(self.output_path, 'log.json')):
                print('restored archived logs')
                with open(os.path.join(self.output_path, 'log.json'), 'r') as fo:
                    self.logs = json.load(fo)
            else:
                print('failed to restore logs, creating new')
                self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)

        if self.output_path == '':
            return

        t = datetime.now().isoformat().split('.')[0]
        epoch_count = len(self.logs)

        if not self.save_log_only:
            K.models.save_model(self.model, os.path.join(self.output_path,
                                                         f'md_e{epoch_count}_t{t}.h5'))

        with open(os.path.join(self.output_path, 'log.json'), 'w+') as fo:
            json.dump(self.logs, fo, default=default, indent=4)

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
            x_np = x_ if isinstance(x_, np.ndarray) else x_.toarray()
            ax.imshow(x_np.T[::-1, :])
        plt.tight_layout()
        res = plt.gcf()
        plt.show()

        if self.output_path != '':
            res.savefig(os.path.join(self.output_path, f'{t}.png'))

        return


class GeneratingOnSameSeedCallback(K.callbacks.Callback):
    def __init__(self, model, sample_generator, pad_length, output_path=''):
        super().__init__()
        self.output_path = output_path
        self.model = model
        self.sample_generator = sample_generator
        self.pad_length = pad_length

    def on_epoch_end(self, epoch, logs={}):
        t = datetime.now().isoformat().split('.')[0]
        sample = self.sample_generator(self.model)

        _, axs = plt.subplots(nrows=4, ncols=4, figsize=(30, 10),
                              subplot_kw={'xticks': [], 'yticks': []})
        for ax, x_ in zip(axs.flat, sample):
            x_np = x_ if isinstance(x_, np.ndarray) else x_.toarray()

            x_padded = np.zeros((self.pad_length, x_np.shape[1]))
            copied_length = min(self.pad_length, x_np.shape[0])
            x_padded[:copied_length, :] = x_np[:copied_length, :]
            ax.imshow(x_padded.T[::-1, :])

        plt.tight_layout()
        res = plt.gcf()
        plt.show()

        if self.output_path != '':
            res.savefig(os.path.join(self.output_path, f'{t}.png'))

        return


class GensimLossPrinter(CallbackAny2Vec):
    def __init__(self):
        self.prev_loss = float("inf")
        self.i = 0

    def on_epoch_end(self, model):
        current_loss = model.get_latest_training_loss()
        print(str(self.i) + " model loss delta:",
              current_loss - self.prev_loss)
        self.prev_loss = current_loss
        self.i += 1
