import unittest
from unittest.case import TestCase
import glob

from left_refference import min_lref_scheme3
import lz77


files = ["Pipfile"]

files = glob.glob("./data/misc/*")
print(files)


class MinLRefTest(TestCase):
    def test_lz77(self):
        for file in files:
            with open(file, "rb") as f:
                text = f.read()
            lz77_solver = min_lref_scheme3(text)
            lz77_true = lz77.encode(text)
            self.assertEqual(len(lz77_solver), len(lz77_true))
            self.assertEqual(text, lz77.decode(lz77_solver))


if __name__ == "__main__":
    unittest.main()
