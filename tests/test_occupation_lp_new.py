import unittest

from best.models.pomdp import POMDP, POMDPNetwork
from best.models.pomdp_sparse_utils import get_T_uxXz, diagonal
from best.solvers.occupation_lp import *
from best.solvers.occupation_lp_new import *
from best.solvers.valiter import solve_reach
import numpy as np

class OccupationLPTest(unittest.TestCase):

  def test_occupation1(self):

    # nondeterministic problem, compare solutions
    network = POMDPNetwork()

    T0 = np.array([[0, 1, 0, 0],
                   [0, 0, 0.5, 0.5],
                   [1, 0, 0, 0],
                   [0, 0, 0, 1]]);

    network.add_pomdp(POMDP([T0], input_names=['a'], state_name='s'))

    T0 = np.array([[1, 0],
                   [0, 1]]);
    T1 = np.array([[0, 1],
                   [0, 1]]);

    network.add_pomdp(POMDP([T0, T1], input_names=['l'], state_name='q'))

    network.add_connection(['s'], 'l', lambda s: {0: set([0]), 1: set([1]), 2: set([0]),  3: set([0])}[s])

    # Define target set
    accept = np.zeros((4,2))
    accept[:,1] = 1

    val_list, pol_list = solve_reach(network, accept)

    P_asS = get_T_uxX(network.pomdps['s'])
    P_lqQ = get_T_uxX(network.pomdps['q'])
    conn = network.connections[0][2]

    reach_prob, _ = occupation_lp_new(P_asS, P_lqQ, conn, s0=0, q0=0, q_target=1)
    np.testing.assert_almost_equal(reach_prob, val_list[0][0, 0])

    reach_prob, _ = occupation_lp_new(P_asS, P_lqQ, conn, s0=1, q0=0, q_target=1)
    np.testing.assert_almost_equal(reach_prob, val_list[0][1, 0])

    reach_prob, _ = occupation_lp_new(P_asS, P_lqQ, conn, s0=2, q0=0, q_target=1)
    np.testing.assert_almost_equal(reach_prob, val_list[0][2, 0])


  def test_occupation2(self):
    # nondeterministic problem without solution
    network = POMDPNetwork()

    T0 = np.array([[0, 1, 0, 0],
                   [0, 0, 0.5, 0.5],
                   [1, 0, 0, 0],
                   [0, 0, 0, 1]]);

    network.add_pomdp(POMDP([T0], input_names=['a'], state_name='s'))

    T0 = np.array([[1, 0],
                   [0, 1]]);
    T1 = np.array([[0, 1],
                   [0, 1]]);

    network.add_pomdp(POMDP([T0, T1], input_names=['l'], state_name='q'))

    network.add_connection(['s'], 'l', lambda s: {0: set([0]), 1: set([0, 1]), 2: set([0]),  3: set([0])}[s])

    # Define target set
    accept = np.zeros((4,2))
    accept[:,1] = 1

    val_list, _ = solve_reach(network, accept)

    P_asS = get_T_uxX(network.pomdps['s'])
    P_lqQ = get_T_uxX(network.pomdps['q'])
    conn = network.connections[0][2]

    reach_prob, _ = occupation_lp_new(P_asS, P_lqQ, conn, s0=0, q0=0, q_target=1)
    np.testing.assert_almost_equal(reach_prob, val_list[0][0, 0])

    reach_prob, _ = occupation_lp_new(P_asS, P_lqQ, conn, s0=1, q0=0, q_target=1)
    np.testing.assert_almost_equal(reach_prob, val_list[0][1, 0])

    reach_prob, _ = occupation_lp_new(P_asS, P_lqQ, conn, s0=2, q0=0, q_target=1)
    np.testing.assert_almost_equal(reach_prob, val_list[0][2, 0])


  def test_occupation3(self):

    network = POMDPNetwork()

    T0 = np.array([[0.5, 0.5], [0, 1]])
    
    network.add_pomdp(POMDP([T0], input_names=['a'], state_name='s'))
    
    T1 = np.array([[0.5, 0.5], [0, 1]])
    T2 = np.array([[1, 0], [0, 1]])

    network.add_pomdp(POMDP([T1, T2], input_names=['l'], state_name='q'))

    network.add_connection(['s'], 'l', lambda s: {0: set([0]), 1: set([0, 1])}[s])

    P_asS = get_T_uxX(network.pomdps['s'])
    P_lqQ = get_T_uxX(network.pomdps['q'])

    accept = np.zeros((2,2))
    accept[:, 1] = 1

    vi_solve, _ = solve_reach(network, accept)

    reach_prob, _ = occupation_lp_new(P_asS, P_lqQ, network.connections[0][2], s0=0, q0=0, q_target=1)
    np.testing.assert_almost_equal(reach_prob, vi_solve[0][0, 0], decimal=5)

    reach_prob, _ = occupation_lp_new(P_asS, P_lqQ, network.connections[0][2], s0=1, q0=0, q_target=1)
    np.testing.assert_almost_equal(reach_prob, vi_solve[0][1, 0], decimal=5)


  def test_occupation4(self):

    network = POMDPNetwork()

    T0 = np.array([[0.5, 0.5], [0, 1]])
    
    network.add_pomdp(POMDP([T0], input_names=['a'], state_name='s'))
    
    T1 = np.array([[0, 1, 0, 0, 0], [0, 0, 0, 0.5, 0.5], [0, 0, 0, 0.3, 0.7], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]])
    T2 = np.array([[0, 0, 1, 0, 0], [0, 0, 0, 0.5, 0.5], [0, 0, 0, 0.3, 0.7], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]])
    T3 = np.array([[0, 0, 0, 1, 0], [0, 0, 0, 1, 0], [0, 0, 0, 1, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]])

    network.add_pomdp(POMDP([T1, T2, T3], input_names=['l'], state_name='q'))

    network.add_connection(['s'], 'l', lambda s: {0: set([0]), 1: set([2])}[s])

    P_asS = get_T_uxX(network.pomdps['s'])
    P_lqQ = get_T_uxX(network.pomdps['q'])

    accept = np.zeros((2,5))
    accept[:, 4] = 1

    vi_solve, _ = solve_reach(network, accept, delta=0.01)

    reach_prob, _ = occupation_lp_new(P_asS, P_lqQ, network.connections[0][2], s0=0, q0=0, q_target=4, delta=0.01)

    np.testing.assert_almost_equal(reach_prob, vi_solve[0][0, 0], decimal=5)


  def test_occupation5(self):

    network = POMDPNetwork()

    T0 = np.array([[0.5, 0.5], [0.1, 0.9]])
    
    network.add_pomdp(POMDP([T0], input_names=['a'], state_name='s'))
    
    T1 = np.array([[0, 1, 0, 0, 0], [0, 0, 0, 0.5, 0.5], [0, 0, 0, 0.3, 0.7], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]])
    T2 = np.array([[0, 0, 1, 0, 0], [0, 0, 0, 0.5, 0.5], [0, 0, 0, 0.3, 0.7], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]])
    T3 = np.array([[0, 0, 0, 1, 0], [0, 0, 0, 1, 0], [0, 0, 0, 1, 0], [0.0, 0, 0, 1, 0], [0, 0, 0, 0, 1]])

    network.add_pomdp(POMDP([T1, T2, T3], input_names=['l'], state_name='q'))

    network.add_connection(['s'], 'l', lambda s: {0: set([0, 1]), 1: set([2])}[s])

    P_asS = get_T_uxX(network.pomdps['s'])
    P_lqQ = get_T_uxX(network.pomdps['q'])

    accept = np.zeros((2,5))
    accept[:, 4] = 1

    vi_solve, _ = solve_reach(network, accept, delta=0.0)

    reach_prob, sol = occupation_lp_new(P_asS, P_lqQ, network.connections[0][2], s0=0, q0=0, q_target=4, delta=0.0)

    np.testing.assert_almost_equal(reach_prob, vi_solve[0][0, 0], decimal=5)
    np.testing.assert_almost_equal(1, 0)


if __name__ == "__main__":
    unittest.main()