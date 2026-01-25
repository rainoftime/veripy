import unittest
from typing import List

import veripy as vp
from veripy import verify


class TestHeapCalls(unittest.TestCase):
    def setUp(self):
        vp.enable_verification()

    def test_heap_mutating_call_updates_caller_heap(self):
        vp.scope('heap_call_mutation')

        @verify(requires=['len(xs) > 0'], ensures=['xs[0] == 1'])
        def g(xs: List[int]) -> int:
            xs[0] = 1
            return 0

        @verify(requires=['len(xs) > 0'], ensures=['xs[0] == 1'])
        def f(xs: List[int]) -> int:
            y = g(xs)
            assert xs[0] == 1
            return y

        vp.verify_all()


if __name__ == "__main__":
    unittest.main()

