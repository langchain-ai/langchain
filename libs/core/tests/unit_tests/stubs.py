class AnyStr(str):
    def __eq__(self, other):
        return isinstance(other, str)
