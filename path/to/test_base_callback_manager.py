# New test file or add to existing test file
import unittest
from path.to.BaseCallbackManager import BaseCallbackManager

class TestBaseCallbackManager(unittest.TestCase):
def test_merge_handlers(self):
manager1 = BaseCallbackManager(handlers=['handler1'], inheritable_handlers=['inheritable1'])
manager2 = BaseCallbackManager(handlers=['handler2'], inheritable_handlers=['inheritable2'])

merged_manager = manager1.merge(manager2)

self.assertEqual(merged_manager.handlers, ['handler1', 'handler2'])
self.assertEqual(merged_manager.inheritable_handlers, ['inheritable1', 'inheritable2'])

def test_merge_duplicate_handlers(self):
manager1 = BaseCallbackManager(handlers=['handler1'], inheritable_handlers=['inheritable1'])
manager2 = BaseCallbackManager(handlers=['handler1'], inheritable_handlers=['inheritable1'])

merged_manager = manager1.merge(manager2)

self.assertEqual(merged_manager.handlers, ['handler1'])
self.assertEqual(merged_manager.inheritable_handlers, ['inheritable1'])

if __name__ == '__main__':
unittest.main()