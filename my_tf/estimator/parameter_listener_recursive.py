from types import SimpleNamespace


class Parameter:
    def __init__(self, *var):
        self.var = None
        self._listeners = []
        for it in var:
            if isinstance(it, Parameter):
                it.add_listener(self)
            else:
                print("输入值为常量")
                self.var = it

    @property
    def value(self):
        if not self.var:
            raise ValueError("值不能为空")
        return self.var

    def _notify(self):
        for listener in self._listeners:
            print(self.__class__.__name__, '----- 通知 ----->>>>', listener.__class__.__name__)
            listener.__update()

    def set_value(self, var=None, **kwargs):
        self.var = var
        print(self.__class__.__name__, 'set_value --> ', self.var)
        self._notify()
        return self

    def _update(self):
        pass

    def __update(self):
        self._update()
        self._notify()

    def add_listener(self, listener):
        self._listeners.append(listener)

    def __repr__(self):
        return f"{self.value}"

# 参数的两种更新方式，自动计算更新，手动设置更新
class ParameterB(Parameter):
    def __init__(self, a: Parameter):
        super().__init__(a)
        self._arg = a

    def _update(self):
        self.var = self._arg.value*2
        print(self.__class__.__name__,'--> ',self.var)


class ParameterC(Parameter):
    def __init__(self, b: Parameter):
        super().__init__(b)
        self._arg = b

    def _update(self):
        self.var = self._arg.value*3
        print(self.__class__.__name__,'--> ',self.var)


class ParameterD(Parameter):
    def __init__(self, a:Parameter,b:Parameter,c:Parameter):
        super().__init__(a,b,c)
        self._arg1 = a
        self._arg2 = b
        self._arg3 = c

    def _update(self):
        self.var = self._arg1.value+self._arg2.value+self._arg3.value
        print(self.__class__.__name__,'--> ',self.var)


# 递归更新，深度优先搜索
class MetaMD(SimpleNamespace):
        a = Parameter(120)
        b = ParameterB(a)
        c = ParameterC(b)
        d = ParameterD(a, b, c)


class FC:
    def __init__(self, var, *args):
        self.var = var
        self._arg = args
        for i in args:
            print(i)
    def show(self):
        print(self._arg.__class__,self._arg)


if __name__ == "__main__":
    MetaMD.a.set_value(12)
    print("=================")
    MetaMD.b.set_value(2323)
    print("=================")

    #fc=FC(12)
    #fc.show()
