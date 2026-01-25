import unittest
from typing import List

import veripy as vp
from veripy import verify


class TestHeapSemantics(unittest.TestCase):
    def setUp(self):
        vp.enable_verification()

    def test_list_aliasing_and_mutation(self):
        vp.scope('heap_list_aliasing')

        @verify(requires=['len(xs) > 0'], ensures=['xs[0] == 1'])
        def mutate_first(xs: List[int]) -> int:
            ys = xs
            xs[0] = 1
            return 0

        vp.verify_all()

    def test_list_oob_is_rejected(self):
        vp.scope('heap_list_oob')

        @verify(requires=['True'], ensures=['True'])
        def mutate_first(xs: List[int]) -> int:
            xs[0] = 1
            return 0

        with self.assertRaises(Exception):
            vp.verify_all()

    def test_div_by_zero_is_rejected(self):
        vp.scope('heap_div0')

        @verify(requires=['True'], ensures=['True'])
        def div(x: int, y: int) -> int:
            return x // y

        with self.assertRaises(Exception):
            vp.verify_all()


if __name__ == "__main__":
    unittest.main()

