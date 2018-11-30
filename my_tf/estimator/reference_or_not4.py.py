
import tensorflow as tf
from types import SimpleNamespace


class Parameter:
    def __init__(self, var):
        self.var = var
        self.auto = True

    def _value(self):
        if isinstance(self.var,self.__class__):
            return self.var.value
        return self.var

    @property
    def value(self):
        if self.auto:
            self.var = self._value()
        if not self.var:
            raise ValueError("值不能为空")
        return self.var

    def set_value(self, var=None, auto=False, **kwargs):
        self.auto = auto
        self.var = var
        print(self.__class__.__name__, 'set value->', self.var)
        return self


class ParaTotalSteps(Parameter):
    def __init__(self, train_size, batch_size, num_epochs):
        self._train_size: Parameter = train_size
        self._batch_size: Parameter = batch_size
        self._num_epochs: Parameter = num_epochs
        super(ParaTotalSteps, self).__init__(self._value())

    def _value(self):
        return (self._train_size.value / self._batch_size.value) *self._num_epochs.value


class MetaMD(SimpleNamespace):
    TRAIN_SIZE = Parameter(1200)  # 为引用对象开辟了空间，并由TRAIN_SIZE指向，下面同理
    BATCH_SIZE = Parameter(20)
    NUM_EPOCHS = Parameter(100)
    TOTAL_STEPS = ParaTotalSteps(TRAIN_SIZE, BATCH_SIZE, NUM_EPOCHS)


    @classmethod
    def HPARAMS(cls):
        return tf.contrib.training.HParams(
            feature_columns=['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth'],
            hidden_units=[10, 10], n_classes=3, learning_rate=0.01,  max_steps=100)



if __name__ == "__main__":
    print(MetaMD.HPARAMS)