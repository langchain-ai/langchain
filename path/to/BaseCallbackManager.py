# File changes (if any)
class BaseCallbackManager:
def __init__(self, handlers=None, inheritable_handlers=None):
self.handlers = handlers if handlers is not None else []
self.inheritable_handlers = inheritable_handlers if inheritable_handlers is not None else []

def merge(self, other):
# Create new lists for merged handlers and inheritable_handlers
merged_handlers = list(set(self.handlers + other.handlers))
merged_inheritable_handlers = list(set(self.inheritable_handlers + other.inheritable_handlers))

# Return a new instance of BaseCallbackManager with the merged lists
return BaseCallbackManager(handlers=merged_handlers, inheritable_handlers=merged_inheritable_handlers)