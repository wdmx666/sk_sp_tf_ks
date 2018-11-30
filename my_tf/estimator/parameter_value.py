from types import SimpleNamespace
import tensorflow as tf


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


class ParameterPath(Parameter):
    def __init__(self, para: Parameter):
        self.parameter = para
        super(ParameterPath, self).__init__(self._value())

    def _value(self):
        value = tf.keras.utils.get_file(self.parameter.value.split('/')[-1], self.parameter.value)
        return value

class ParameterTest(Parameter):
    def __init__(self, cls):
        self.cls = cls
        super().__init__(self._value())

    def _value(self):
        value = self.cls.train_url.value+"   +  " +self.cls.test_url.value
        print(value)
        return value


# 采用Parameter的方式主要是为了包装减少set、get模板代码
class MetaRaw(SimpleNamespace):
    train_url = Parameter("http://download.tensorflow.org/data/iris_training.csv")
    train_path = ParameterPath(train_url)
    test_url = Parameter("http://download.tensorflow.org/data/iris_test.csv")
    test_path = ParameterPath(test_url)
    feature_names = Parameter(['sepallength', 'sepalwidth', 'petallength', 'petalwidth'])
    numeric_feature_names = Parameter(feature_names)


if __name__ == "__main__":
    v=MetaRaw.train_path.value

    v2 = MetaRaw.numeric_feature_names.value
    print(v,v2)
    print("===============================")
    MetaRaw.train_path.set_value("http://download.tensorflow.org/data/iris_test.csv")
    v = MetaRaw.train_path.value
    v2 = MetaRaw.numeric_feature_names.value
    print(v,v2)
    print("=============================================")
    MetaRaw.train_path.set_value(auto=True)
    v = MetaRaw.train_path.value
    v2 = MetaRaw.numeric_feature_names.value
    print(v,v2)
