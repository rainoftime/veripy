import unittest

import veripy as vp
from veripy import verify


class TestSoundnessRegressions(unittest.TestCase):
    def setUp(self):
        vp.enable_verification()

    def test_len_on_int_is_rejected(self):
        vp.scope('sound_len_on_int')

        @verify(requires=['True'], ensures=['ans == 1'])
        def f(x: int) -> int:
            return len(x)  # Python would raise TypeError; verifier must reject too.

        with self.assertRaises(Exception) as ctx:
            vp.verify_all()
        self.assertIn('len expects a list or dict', str(ctx.exception))

    def test_if_branch_does_not_drop_statements(self):
        vp.scope('sound_if_branch_full')

        @verify(requires=['x > 0'], ensures=['ans > 0'])
        def f(x: int) -> int:
            ans = 1
            if x > 0:
                ans = 1
                ans = -1  # Should make postcondition fail when x > 0.
            return ans

        with self.assertRaises(Exception):
            vp.verify_all()


if __name__ == "__main__":
    unittest.main()
