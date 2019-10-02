import unittest

from best.models.pomdp import POMDP, POMDPNetwork
from best.models.pomdp_sparse_utils import get_T_uxXz, diagonal
from best.solvers.occupation_lp import *
from best.solvers.valiter import solve_reach
import numpy as np

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
    import numpy as np
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

  def test_demo(self):
    from Demos.demo_Models import simple_robot
    import polytope as pc
    import numpy as np

    Robot = simple_robot()

    Robot.input_space = pc.box2poly(np.kron(np.ones((Robot.m, 1)), np.array([[-1, 1]])))  # continuous set of inputs
    Robot.state_space = pc.box2poly(np.kron(np.ones((Robot.dim, 1)), np.array([[-10, 10]])))  # X space

    print("-------Grid the robots state space------")
    #  set the state spaces:
    from Reduce.Gridding import grid as griddd

    d_opt = np.array([[0.69294], [0.721]])  # tuned gridding ratio from a previous paper
    d = 1.6 * d_opt  # with distance measure 0.6=default
    un = 3

    Ms, srep = griddd(Robot, d, un=un)

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
    Ms._state_space[-1] = (np.nan, np.nan)
    # (Ms._state_space[s][0] > 4)

    network.add_connection(['s'], 'l', lambda s: set([1]) if (Ms._state_space[s][0] > 4) else set([0]))
    accept = np.zeros((T00.shape[1], 2))
    accept[:, 1] = 1

    val_list, pol_list = solve_reach(network, accept)
    P_asS = diagonal(get_T_uxXz(network.pomdps['s']), 2, 3)
    P_lqQ = diagonal(get_T_uxXz(network.pomdps['q']), 2, 3)
    conn = network.connections[0][2]
    delt = []
    reach = []
    val = []
    for delta in np.linspace(0.0001, 0.05, 5):
      t = time.time()
      delt += [delta]
      reach_prob, val2 = solve_delta(P_asS, P_lqQ, conn, delta, s0=0, q0=0, q_target=1)
      reach += [reach_prob]
      val += [np.sum(val2, axis=0)]
      print(time.time() - t)
      # print(val2)
    # np.testing.assert_almost_equal(reach_prob, val_list[0][0, 0],decimal=5)
    print(reach)

  def test_demo_scaling(self):
    t_list = []
    delt = []
    reach = []
    val = []
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
    for factor in [1.8,1.7,1.6,1.5,1.4]:
      d = factor * d_opt  # with distance measure 0.6=default
      print("factor",factor)
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

      t=time.time()
      delta = 0.01
      delt += [delta]
      reach_prob, val2 = solve_delta(P_asS, P_lqQ, conn, delta, s0=0, q0=0, q_target=1)
      reach += [reach_prob]
      val += [np.sum(val2, axis=0)]
      t_list += [time.time()-t]
      print(time.time()-t)

      # print(val2)
    # np.testing.assert_almost_equal(reach_prob, val_list[0][0, 0],decimal=5)
    print("reach prob",reach)
    print("solve time ",t_list)


  def test_basic(self):
      # test lp that pushes robot to  sink state = leaving the gridded area
      from Demos.demo_Models import simple_robot
      import polytope as pc

      Robot = simple_robot()

      Robot.input_space = pc.box2poly(np.kron(np.ones((Robot.m, 1)), np.array([[-1, 1]])))  # continuous set of inputs
      Robot.state_space = pc.box2poly(np.kron(np.ones((Robot.dim, 1)), np.array([[-10, 10]])))  # X space

      print("-------Grid the robots state space------")
      #  set the state spaces:
      from Reduce.Gridding import grid as griddd


      d_opt = np.array([[0.69294], [0.721]])  # tuned gridding ratio from a previous paper
      d = 1.4 * d_opt  # with distance measure 0.6=default
      un = 3

      Ms, srep  = griddd(Robot, d, un=un)

      T00 = Ms.transition

      t = time.time()
      delta = 0.01
      print(np.max(np.sum(T00, axis=2)))
      model,sol = solve_pure_occ(T00.tolist(), len(T00[0])-1, 0, delta)

      Tlist = T00.tolist()
      print(time.time()-t)

        # print(val2)
      # np.testing.assert_almost_equal(reach_prob, val_list[0][0, 0],decimal=5)


  def test_basic_v2(self):
      T00 = np.array([[[0, 1, 0, 0],
                     [0, 0, 0.5, 0.5],
                     [1, 0, 0, 0],
                     [0, 0, 0, 1]]]);

      t = time.time()
      delta = 0.01
      print(np.sum(T00, axis=2))
      print('var to max', len(T00[0])-1)
      model,sol = solve_pure_occ(T00.tolist(), [len(T00[0])-1], 0, delta)

      Tlist = T00.tolist()
      print(time.time()-t)

      print(np.argmax(sol['x']))
      print(min(sol['x']))


if __name__ == "__main__":
    unittest.main()