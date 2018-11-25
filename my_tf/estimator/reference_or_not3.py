
import tensorflow as tf
from types import SimpleNamespace

class Parameter:
    def __init__(self, var):
        self._var = var

    @property
    def value(self):
        return self._var

    def set_value(self, var):
        self._var = var
        return self

    def __repr__(self):
        return f"{self.value}"


class MetaMD(SimpleNamespace):
    def __init__(self):
        self.TRAIN_SIZE = 1200

        self.NUM_EPOCHS = 100
        self.BATCH_SIZE = 20
        self.TOTAL_STEPS = (self.TRAIN_SIZE /self. BATCH_SIZE) * self.NUM_EPOCHS


        self.HPARAMS = tf.contrib.training.HParams(
            feature_columns=['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth'],
            hidden_units=[10, 10], n_classes=3, learning_rate=0.01,  max_steps=self.TOTAL_STEPS)


def showMeta():
    print(MetaMD.TOTAL_STEPS)
    print(MetaMD.HPARAMS)
    print('-------------------')
    print(id(MetaMD.TOTAL_STEPS), MetaMD.TOTAL_STEPS)
    print("=====================================================")


def show(md):
    print(md.TOTAL_STEPS)
    print(md.HPARAMS)
    print('-------------------')
    print(id(md.TOTAL_STEPS),md.TOTAL_STEPS)
    print("=====================================================")


def change_meta(md):
    #MetaMD.TOTAL_STEPS = 4586
    #md.TOTAL_STEPS = 2121
    md.TOTAL_STEPS =2121

if __name__ == "__main__":
    md = MetaMD()
    show(md)
    change_meta(md)
    show(md)