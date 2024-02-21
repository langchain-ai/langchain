import importlib

class LazyImport(object):
    """Lazy import python module till use.

    Args:
        module_name (string): The name of module imported later
    """

    def __init__(self, module_name):
        """The module name initialization."""
        self.module_name = module_name
        self.module = None

    def __getattr__(self, name):
        """The __getattr__ function."""
        try:
            self.module = importlib.import_module(self.module_name)
            mod = getattr(self.module, name)
        except:
            spec = importlib.util.find_spec(str(self.module_name + '.' + name))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        return mod

    def __call__(self, *args, **kwargs):
        """The __call__ function."""
        function_name = self.module_name.split('.')[-1]
        module_name = self.module_name.split(f'.{function_name}')[0]
        self.module = importlib.import_module(module_name)
        function = getattr(self.module, function_name)
        return function(*args, **kwargs)
