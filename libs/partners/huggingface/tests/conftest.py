import sys
import os

package_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'huggingface'))
sys.path.insert(0, package_path)