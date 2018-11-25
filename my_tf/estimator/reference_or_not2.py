
import tensorflow as tf
from types import SimpleNamespace

#  #####################数据ETL##########################


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


class MetaRaw(SimpleNamespace):
    TARGET = Parameter('Species')
    FEATURE_NAMES = Parameter(['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth'])


# class MetaMD(SimpleNamespace):
#     TRAIN_SIZE = Parameter(1200)
#
#     NUM_EPOCHS = Parameter(100)
#     BATCH_SIZE = Parameter(20)
#     TOTAL_STEPS = Parameter((TRAIN_SIZE.value / BATCH_SIZE.value) * NUM_EPOCHS.value)
#
#     HPARAMS = tf.contrib.training.HParams(
#         feature_columns=['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth'],
#         hidden_units=[10, 10],
#         n_classes=3,
#         learning_rate=0.01,
#         max_steps=TOTAL_STEPS.value)


class MetaMD(SimpleNamespace):
    TRAIN_SIZE: Parameter = Parameter(1200)

    NUM_EPOCHS = 100
    BATCH_SIZE = 20
    @property
    def TOTAL_STEPS(self):
        return Parameter((self.TRAIN_SIZE.value / self.BATCH_SIZE) * self.NUM_EPOCHS)
        #self.TOTAL_STEPS = (self.TRAIN_SIZE / self.BATCH_SIZE) * self.NUM_EPOCHS

    @property
    def HPARAMS(self):
        # 这样确保每次拿到的都是最新状态下，装填形成的值，这样该属性像监听了其各种依赖一样；
        # 而且每种对象的更新方式有所不同，不都是像键值对那样更新，或许你可以绕弯这么做；通过
        # 将依赖(输入)通过函数参数输入，返回得到新的参数，能够保证A在更新之后用的是新的A状态值，
        # 因为输入就是先于输出，输出依赖输入。
        return tf.contrib.training.HParams(
            feature_columns=['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth'],
            hidden_units=[10, 10], n_classes=3, learning_rate=0.01, max_steps=self.TOTAL_STEPS)


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
    showMeta()
    show(md)
    change_meta(md)
    showMeta()
    show(md)