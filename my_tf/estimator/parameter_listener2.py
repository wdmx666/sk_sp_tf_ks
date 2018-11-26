from types import SimpleNamespace

class Parameter:
    def __init__(self, var):
        self.var = var
        self._listeners=[]

    @property
    def value(self):
        if not self.var:
            raise ValueError("值不能为空")
        return self.var

    def set_value(self, var=None, **kwargs):
        self.var = var
        print(self.__class__.__name__, 'set_value --> ', self.var)
        for listener in self._listeners:
            listener.update()

        return self

    def update(self):
        pass

    def add_listener(self, listener):
        self._listeners.append(listener)

    def __repr__(self):
        return f"{self.value}"

# 参数的两种更新方式，自动计算更新，手动设置更新
class ParameterB(Parameter):
    def __init__(self, a: Parameter):
        super().__init__(None)
        self._arg = a
    def update(self):
        self.var = self._arg.value*2
        print(self.__class__.__name__,'--> ',self.var)
        for it in self._listeners:
            it.update()


class ParameterC(Parameter):
    def __init__(self, b: Parameter):
        super().__init__(None)
        self._arg = b

    def update(self):
        self.var = self._arg.value*3
        print(self.__class__.__name__,'--> ',self.var)
        for it in self._listeners:
            it.update()


class ParameterD(Parameter):
    def __init__(self,a:Parameter,b:Parameter,c:Parameter):
        super().__init__(None)
        self._arg1 = a
        self._arg2 = b
        self._arg3 = c
    def update(self):
        self.var = self._arg1.value+self._arg2.value+self._arg3.value
        print(self.__class__.__name__,'--> ',self.var)
        for it in self._listeners:
            it.update()

class MetaMD(SimpleNamespace):
        a = Parameter(120)
        b = ParameterB(a)
        c = ParameterC(b)
        d = ParameterD(a, b, c)
        a.add_listener(b)
        b.add_listener(c)
        a.add_listener(d)
        #c.add_listener(d)

if __name__ == "__main__":
    MetaMD.a.set_value(12)
    print("=================")
    MetaMD.b.set_value(2323)
