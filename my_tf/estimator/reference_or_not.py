from types import SimpleNamespace


class Parameter:
    def __init__(self, var):
        self._var = var
        self._listener = []

    @property
    def value(self):
        return self._var

    def set_value(self, var):
        self._var = var
        self._notify()
        return self

    def add_listener(self, fn):
        self._listener.append(fn)
        return self

    def _notify(self):
        for it in self._listener:
            it()

    def __repr__(self):
        return f"{self.value}"


class MetaRaw(SimpleNamespace):
    TARGET = 'Species'
    FEATURE_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']
    FEATURE_NAMES2 = FEATURE_NAMES

    @classmethod
    def updateFEATURE_NAMES(cls, value):
        cls.FEATURE_NAMES = value
        cls.updateFEATURE_NAMES2(value)

    @classmethod
    def updateFEATURE_NAMES2(cls, value):
        cls.FEATURE_NAMES2 = value


def show():
    print(MetaRaw.FEATURE_NAMES)
    print(MetaRaw.FEATURE_NAMES2)



def change_meta():
    MetaRaw.updateFEATURE_NAMES(list('abcd'))


if __name__ == "__main__":
    show()
    change_meta()
    show()