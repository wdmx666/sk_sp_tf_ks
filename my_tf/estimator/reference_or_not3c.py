
import tensorflow as tf
from types import SimpleNamespace

class MetaMD(SimpleNamespace):
    TRAIN_SIZE = 1200

    NUM_EPOCHS = 100
    BATCH_SIZE = 20
    TOTAL_STEPS = (TRAIN_SIZE / BATCH_SIZE) * NUM_EPOCHS
    HPARAMS = tf.contrib.training.HParams(
        feature_columns=['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth'],
        hidden_units=[10, 10], n_classes=3, learning_rate=0.01,  max_steps=TOTAL_STEPS)

    @classmethod
    def set_attr(cls,k,v):
        if hasattr(cls,k):
            print(k,v)

    @classmethod
    def get_NUM_EPOCHS(cls):
        return cls.TRAIN_SIZE

class Meta(object):
    def __init__(self):
        self._train_size=10

    @property
    def train_size(self):
        return self._train_size

def showMeta():
    print(MetaMD.TOTAL_STEPS)
    print(MetaMD.HPARAMS)
    print('-------------------')
    print(id(MetaMD.TOTAL_STEPS), MetaMD.TOTAL_STEPS)
    print("=====================================================")



def change_meta(md):
    #MetaMD.TOTAL_STEPS = 4586
    #md.TOTAL_STEPS = 2121
    md.TOTAL_STEPS =2121

if __name__ == "__main__":
    print(dir(MetaMD))
    mt =Meta()
   # MetaMD.set_attr('NUM_EPOCHS',121212)
    print(MetaMD.get_NUM_EPOCHS())
    print(mt.train_size)