import numpy as np
from best.logic.translate import *

import unittest
class TEST_LOGIC(unittest.TestCase):

  def test_logic(self):
    pm = formula_to_logic('a & b')

    np.testing.assert_almost_equal(pm.T((0,0)).todense(), np.array([[1,0], [1,0]]))
    np.testing.assert_almost_equal(pm.T((1,0)).todense(), np.array([[1,0], [1,0]]))
    np.testing.assert_almost_equal(pm.T((0,1)).todense(), np.array([[1,0], [1,0]]))
    np.testing.assert_almost_equal(pm.T((1,1)).todense(), np.array([[0,1], [0,1]]))

    pm = formula_to_logic('a | b')

    np.testing.assert_almost_equal(pm.T((0,0)).todense(), np.array([[1,0], [1,0]]))
    np.testing.assert_almost_equal(pm.T((1,0)).todense(), np.array([[0,1], [0,1]]))
    np.testing.assert_almost_equal(pm.T((0,1)).todense(), np.array([[0,1], [0,1]]))
    np.testing.assert_almost_equal(pm.T((1,1)).todense(), np.array([[0,1], [0,1]]))

    pm = formula_to_logic('a & b & c')

    np.testing.assert_almost_equal(pm.T((0,0,0)).todense(), np.array([[1,0], [1,0]]))
    np.testing.assert_almost_equal(pm.T((1,0,1)).todense(), np.array([[1,0], [1,0]]))
    np.testing.assert_almost_equal(pm.T((0,1,0)).todense(), np.array([[1,0], [1,0]]))
    np.testing.assert_almost_equal(pm.T((1,1,1)).todense(), np.array([[0,1], [0,1]]))


if __name__ == "__main__":
    unittest.main()