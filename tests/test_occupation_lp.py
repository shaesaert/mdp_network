import numpy as np
import sparse
from best.models.pomdp import POMDP, POMDPNetwork
from best.solvers.occupation_lp import *
from best.solvers.valiter import *

import unittest


class DemoTestCase2(unittest.TestCase):
  """Demo test case 2."""

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

    P_asS = diagonal(get_T_uxXz(network.pomdps['s']), 2, 3)
    P_lqQ = diagonal(get_T_uxXz(network.pomdps['q']), 2, 3)
    conn = network.connections[0][2]

    reach_prob, _ = solve_robust(P_asS, P_lqQ, conn, s0=0, q0=0, q_target=1)
    np.testing.assert_almost_equal(reach_prob, val_list[0][0, 0])

    reach_prob, _ = solve_exact(P_asS, P_lqQ, conn, s0=0, q0=0, q_target=1)
    np.testing.assert_almost_equal(reach_prob, val_list[0][0, 0])


    reach_prob, _ = solve_robust(P_asS, P_lqQ, conn, s0=1, q0=0, q_target=1)
    np.testing.assert_almost_equal(reach_prob, val_list[0][1, 0])

    reach_prob, _ = solve_exact(P_asS, P_lqQ, conn, s0=1, q0=0, q_target=1)
    np.testing.assert_almost_equal(reach_prob, val_list[0][1, 0])


    reach_prob, _ = solve_robust(P_asS, P_lqQ, conn, s0=2, q0=0, q_target=1)
    np.testing.assert_almost_equal(reach_prob, val_list[0][2, 0])

    reach_prob, _ = solve_exact(P_asS, P_lqQ, conn, s0=2, q0=0, q_target=1)
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

    val_list, pol_list = solve_reach(network, accept)
    print(get_T_uxXz(network.pomdps['s']).todense())
    P_asS = diagonal(get_T_uxXz(network.pomdps['s']), 2, 3)
    P_lqQ = diagonal(get_T_uxXz(network.pomdps['q']), 2, 3)
    conn = network.connections[0][2]

    reach_prob, _ = solve_robust(P_asS, P_lqQ, conn, s0=0, q0=0, q_target=1)
    np.testing.assert_almost_equal(reach_prob, val_list[0][0, 0])

    reach_prob, _ = solve_robust(P_asS, P_lqQ, conn, s0=1, q0=0, q_target=1)
    np.testing.assert_almost_equal(reach_prob, val_list[0][1, 0])

    reach_prob, _ = solve_robust(P_asS, P_lqQ, conn, s0=2, q0=0, q_target=1)
    np.testing.assert_almost_equal(reach_prob, val_list[0][2, 0])

  def test_ltl_synth(self):
    delta: object = 0.
    T1 = np.array([[0.25, 0.25, 0.25, 0.25], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    T2 = np.array([[0, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0], [0.9, 0, 0, 0.1]])

    system = POMDP([T1, T2], state_name='x')

    network = POMDPNetwork([system])

    formula = '( ( F s1 ) & ( F s2 ) )'

    predicates = {'s1': (['x'], lambda x: set([int(x==1)])),
             's2': (['x'], lambda x: set([int(x==3)]))}

    dfsa, dfsa_init, dfsa_final = formula_to_pomdp(formula)

    network_copy = copy.deepcopy(network)

    network_copy.add_pomdp(dfsa)
    print(network_copy)
    for ap, (outputs, conn) in predicates.items():
      if ap in dfsa.input_names:
        network_copy.add_connection(outputs, ap, conn)

    Vacc = np.zeros(network_copy.N)
    Vacc[..., list(dfsa_final)[0]] = 1


    val, pol = solve_reach(network_copy, Vacc, delta=delta)
    P_asS = diagonal(get_T_uxXz(network_copy.pomdps['x']), 2, 3)
    P_lqQ = diagonal(get_T_uxXz(network_copy.pomdps['mu']), 2, 3)
    conn = network_copy.connections[0][2]

    reach_prob, _ = solve_robust(P_asS, P_lqQ, conn, s0=0, q0=list(dfsa_init)[0], q_target=list(dfsa_final)[0])

    print(reach_prob)
    np.testing.assert_almost_equal(pol.val[0][:,0], [0.5, 0, 0, 0.5], decimal=4)

    np.testing.assert_almost_equal(reach_prob,val[0,0], decimal=9)


  def test_demo(self):
    from Demos.demo_Models import simple_robot
    import polytope as pc
    import numpy as np


    Robot = simple_robot()

    Robot.input_space = pc.box2poly(np.kron(np.ones((Robot.m, 1)), np.array([[-1, 1]])))  # continuous set of inputs
    Robot.state_space = pc.box2poly(np.kron(np.ones((Robot.dim, 1)), np.array([[-10, 10]])))  # X space


    from Demos.Robot_navigation import specify_robot
    from Controller.Specifications import Fsa

    # specify required behaviour
    formula, regions = specify_robot()  # type: dict

    fsaform = Fsa()  # this is a FSA , contains a digraph
    fsaform.from_formula(formula)  # given a more complex formula


    print("The synthesised formula is:", formula)

    print("-------Grid the robots state space------")
    #  set the state spaces:
    from Reduce.Gridding import grid as griddd


    d_opt = np.array([[0.69294], [0.721]])  # tuned gridding ratio from a previous paper
    d = 2 * d_opt  # with distance measure 0.6=default
    un = 3

    Ms, srep  = griddd(Robot, d, un=un)

    print('# states')
    print(len(Ms._state_space))
    network = POMDPNetwork()

    T00 = Ms.transition

    network.add_pomdp(POMDP(T00, input_names=['a'], state_name='s'))

    T0 = np.array([[1, 0],
                   [0, 1]]);
    T1 = np.array([[0, 1],
                   [0, 1]]);
    network.add_pomdp(POMDP([T0, T1], input_names=['l'], state_name='q'))
    Ms._state_space[-1] = (np.nan,np.nan)
    # (Ms._state_space[s][0] > 4)

    network.add_connection(['s'], 'l', lambda s: set([1]) if (Ms._state_space[s][0] > 4) else set([0]) )
    accept = np.zeros((T00.shape[1],2))
    accept[:,1] = 1

    val_list, pol_list = solve_reach(network, accept)
    P_asS = diagonal(get_T_uxXz(network.pomdps['s']), 2, 3)
    P_lqQ = diagonal(get_T_uxXz(network.pomdps['q']), 2, 3)
    conn = network.connections[0][2]
    delt = []
    reach = []
    val = []
    for delta in np.linspace(0.0001, 0.05, 5):
      t=time.time()
      delt += [delta]
      reach_prob, val2 = solve_delta(P_asS, P_lqQ, conn, delta, s0=0, q0=0, q_target=1)
      reach += [reach_prob]
      val += [np.sum(val2, axis=0)]
      print(time.time()-t)
      # print(val2)
    # np.testing.assert_almost_equal(reach_prob, val_list[0][0, 0],decimal=5)
    print(reach)



if __name__ == "__main__":
    unittest.main()