import importlib
import os
import sys
import unittest


class VercelImportTest(unittest.TestCase):
    def test_app_imports_without_loading_camera_or_detector_on_vercel(self):
        os.environ["VERCEL"] = "1"
        for module_name in ["app", "camera", "detector"]:
            sys.modules.pop(module_name, None)

        app_module = importlib.import_module("app")

        self.assertTrue(hasattr(app_module, "app"))
        self.assertNotIn("camera", sys.modules)
        self.assertNotIn("detector", sys.modules)


if __name__ == "__main__":
    unittest.main()
