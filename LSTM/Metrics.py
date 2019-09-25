import numpy as np
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []
        self.epoch = 0

    def on_epoch_end(self, epoch, logs={}):
        if 'parallel' in self.file_name:
            # assume sigmoid function used in the final activation value
            val_predict = np.where(self.model.predict(self.validation_data[0], batch_size=32) > 0.5, 1, 0)
        else:
            val_predict = np.asarray(self.model.predict_classes(self.validation_data[0], batch_size=32))
        val_targ = self.validation_data[1]
        _val_f1 = f1_score(val_targ, val_predict)
        _val_precision = precision_score(val_targ, val_predict)
        _val_recall = recall_score(val_targ, val_predict)
        self.val_f1s.append(_val_f1)
        self.val_precisions.append(_val_precision)
        self.val_recalls.append(_val_recall)
        print(' — val_f1: %f — val_precision: %f — val_recall %f' % (_val_f1, _val_precision, _val_recall))
        print('%2d,%f,%f,%f' % (self.epoch, _val_f1, _val_precision, _val_recall), file=open(self.file_name, 'a'))
        self.epoch += 1
        return

    def set_file_name(self, file_name):
        self.file_name = file_name
