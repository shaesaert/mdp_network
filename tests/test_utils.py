import scipy.sparse as sp
import numpy as np
import unittest

from best.utils import sparse_tensordot

class TEST_UTILS(unittest.TestCase):

	def test_sparse1(self):

		a = sp.coo_matrix(np.random.randn(5, 9))
		b = np.random.randn(9)

		c = sparse_tensordot(a, b, 0)

		np.testing.assert_equal(c, a.dot(b))

	def test_sparse2(self):

		a = sp.coo_matrix(np.random.randn(5, 8))
		b = np.random.randn(9,8,7)

		c = sparse_tensordot(a, b, 1)

		np.testing.assert_equal(c.shape, (9,5,7))

		for i in range(9):
			for j in range(7):
				np.testing.assert_equal(c[i,:,j], a.dot(b[i,:,j]))

	def test_sparse3(self):

		a = sp.coo_matrix(np.random.randn(5, 7))
		b = np.random.randn(9,8,7)

		c = sparse_tensordot(a, b, 2)

		np.testing.assert_equal(c.shape, (9,8,5))

		for i in range(9):
			for j in range(8):
				np.testing.assert_equal(c[i,j,:], a.dot(b[i,j,:]))


	def test_sparse4(self):

		a = sp.coo_matrix(np.random.randn(5, 9))
		b = np.random.randn(9,8,7)

		c = sparse_tensordot(a, b, 0)

		np.testing.assert_equal(c.shape, (5,8,7))

		for i in range(8):
			for j in range(7):
				np.testing.assert_equal(c[:,i,j], a.dot(b[:,i,j]))


if __name__ == "__main__":
    unittest.main()