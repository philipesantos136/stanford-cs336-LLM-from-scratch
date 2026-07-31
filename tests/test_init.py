import os
import unittest

class TestRepositoryStructure(unittest.TestCase):
    def test_repository_structure(self):
        self.assertTrue(os.path.exists("README.md"))
        self.assertTrue(os.path.exists(".gitignore"))
        self.assertTrue(os.path.exists("docs/adr/0001-repository-initialization.md"))

if __name__ == "__main__":
    unittest.main()
