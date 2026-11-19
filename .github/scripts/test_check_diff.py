import unittest

from check_diff import _get_configs_for_single_dir


class TestPythonVersions(unittest.TestCase):
    def test_core_runs_full_matrix(self) -> None:
        configs = _get_configs_for_single_dir("test", "libs/core")
        self.assertEqual(
            [config["python-version"] for config in configs],
            ["3.11", "3.12", "3.13", "3.14"],
        )

    def test_other_packages_run_min_and_max(self) -> None:
        for package in ("openai", "perplexity", "anthropic"):
            for job in ("test", "lint", "compile-integration-tests"):
                with self.subTest(package=package, job=job):
                    configs = _get_configs_for_single_dir(
                        job, f"libs/partners/{package}"
                    )
                    self.assertEqual(
                        [config["python-version"] for config in configs],
                        ["3.11", "3.14"],
                    )
