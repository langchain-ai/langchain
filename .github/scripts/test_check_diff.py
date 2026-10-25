import unittest

from check_diff import _get_configs_for_single_dir


class TestIntegrationPythonVersions(unittest.TestCase):
    def test_pillow_integration_jobs_require_python_311(self) -> None:
        for package in ("openai", "perplexity"):
            for job in ("compile-integration-tests", "lint"):
                with self.subTest(package=package, job=job):
                    configs = _get_configs_for_single_dir(
                        job, f"libs/partners/{package}"
                    )
                    self.assertEqual(
                        [config["python-version"] for config in configs],
                        ["3.11", "3.14"],
                    )

    def test_unit_tests_keep_python_310(self) -> None:
        for package in ("openai", "perplexity"):
            with self.subTest(package=package):
                configs = _get_configs_for_single_dir(
                    "test", f"libs/partners/{package}"
                )
                self.assertEqual(
                    [config["python-version"] for config in configs], ["3.10", "3.14"]
                )

    def test_other_integrations_keep_python_310(self) -> None:
        configs = _get_configs_for_single_dir(
            "compile-integration-tests", "libs/partners/anthropic"
        )
        self.assertEqual(
            [config["python-version"] for config in configs], ["3.10", "3.14"]
        )
