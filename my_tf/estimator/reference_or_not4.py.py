
import tensorflow as tf
from types import SimpleNamespace


class Parameter:
    def __init__(self, var):
        self.var = var

    @property
    def value(self):
        if not self.var:
            raise ValueError("值不能为空")
        return self.var

    def set_value(self, var=None, **kwargs):
        self.var = var
        return self

    def __repr__(self):
        return f"{self.value}"


class TotalSteps(Parameter):
    def __init__(self, train_size: Parameter, batch_size: Parameter, num_epochs: Parameter):
        super().__init__(None)
        self._train_size = train_size
        self._batch_size = batch_size
        self._num_epochs = num_epochs
        self._auto = True

    @property
    def value(self):
        if self._auto:
            return (self._train_size.value / self._batch_size.value) * self._num_epochs.value
        else:
            return self.var

    def set_value(self, var, auto=True):
        self._auto = auto
        self.var = var


class MetaMD(SimpleNamespace):
        TRAIN_SIZE = Parameter(1200)  # 为引用对象开辟了空间，并由TRAIN_SIZE指向，下面同理
        BATCH_SIZE = Parameter(20)
        NUM_EPOCHS = Parameter(100)
        TOTAL_STEPS = TotalSteps(TRAIN_SIZE, BATCH_SIZE, NUM_EPOCHS)
        HPARAMS = tf.contrib.training.HParams = tf.contrib.training.HParams(
                feature_columns=['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth'],
                hidden_units=[10, 10], n_classes=3, learning_rate=0.01,  max_steps=100)


def showMeta():
    print(MetaMD.TOTAL_STEPS.value)
    print(MetaMD.TRAIN_SIZE.value)
    print(MetaMD.HPARAMS.value)
    print('-------------------')
    print(id(MetaMD.TOTAL_STEPS), MetaMD.TOTAL_STEPS)
    print("=====================================================")

def show(md):
    print(md.TOTAL_STEPS)
    print(md.HPARAMS)
    print('-------------------')
    print(id(md.TOTAL_STEPS),md.TOTAL_STEPS)
    print("=====================================================")


def change_meta():
    MetaMD.TOTAL_STEPS.set_value(4586, auto=False)
    print(MetaMD.HPARAMS.set_value(max_steps=1200))
    #md.TOTAL_STEPS = 2121
    #md.TOTAL_STEPS =2121

if __name__ == "__main__":
    showMeta()
    change_meta()
    showMeta()