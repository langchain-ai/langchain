import sys
from pathlib import Path

# Add the scripts directory to the path
scripts_dir = Path(__file__).parent.parent.parent / "scripts"
sys.path.insert(0, str(scripts_dir))

from get_min_versions import _check_python_version_from_requirement
from packaging.requirements import Requirement


def test_check_python_version_with_marker():
    """Test that python_version markers are correctly detected"""
    # This should return True because it checks the version
    req = Requirement("some-package; python_version >= '3.8'")
    result = _check_python_version_from_requirement(req, "3.9")
    assert result is True
    

def test_check_python_version_without_marker():
    """Test that requirements without version markers return True"""
    req = Requirement("some-package")
    result = _check_python_version_from_requirement(req, "3.9")
    assert result is True


def test_check_python_version_with_non_version_marker():
    """Test that non-python-version markers return True (the bug case)"""
    # This tests the bug - markers like sys_platform should return True
    # without attempting version checks
    req = Requirement("some-package; sys_platform == 'linux'")
    result = _check_python_version_from_requirement(req, "3.9")
    assert result is True