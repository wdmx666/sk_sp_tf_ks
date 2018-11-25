

from types import SimpleNamespace
class Parameter:
    def __init__(self, var):
        self.var = var

    @property
    def value(self):
        if not self.var:
            raise ValueError("值不能为空")
        return self.var

    def set_value(self, var, **kwargs):
        self.var = var
        return self

    def __repr__(self):
        return f"{self.value}"


class MetaMD:
    def __init__(self):
        self.TRAIN_SIZE = 1200
        self.BATCH_SIZE = 20
        self.BATCH_SIZE2 = self.BATCH_SIZE


class MetaRaw:
    TARGET = Parameter('Species')
    FEATURE_NAMES = Parameter(['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth'])
    CSV_COLUMN_NAMES = Parameter(FEATURE_NAMES.value + [TARGET.value])


class MetaRaw2:
    TARGET = 'Species'
    FEATURE_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']

    @property
    @classmethod
    def CSV_COLUMN_NAMES(cls):
        return cls.FEATURE_NAMES + [cls.TARGET]


def showMeta():
    print(MetaRaw2.TARGET)
    print(MetaRaw2.CSV_COLUMN_NAMES)

    print("=====================================================")

def changeMeta():
    MetaRaw2.TARGET="abc"

    print("==================+++++++++++++++++===========================")

def show(md):
    print(md.TRAIN_SIZE, md.BATCH_SIZE, md.BATCH_SIZE2)
    print('-------------------')
    print(id(md.BATCH_SIZE), md.BATCH_SIZE)
    print(MetaRaw.TARGET.value)
    print("=====================================================")


def change_meta(md):
    #MetaMD.TOTAL_STEPS = 4586
    #md.TOTAL_STEPS = 2121
    md.BATCH_SIZE = 2121

if __name__ == "__main__":
    # md = MetaMD()
    # show(md)
    # change_meta(md)
    # show(md)
    showMeta()
    changeMeta()
    showMeta()