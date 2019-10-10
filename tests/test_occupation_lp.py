import unittest

from best.models.pomdp import POMDP, POMDPNetwork
from best.models.pomdp_sparse_utils import get_T_uxXz, diagonal
from best.solvers.occupation_lp import *
from best.solvers.occupation_lp_new import *
from best.solvers.valiter import solve_reach
import numpy as np
import copy

class MDP1DFA1(unittest.TestCase):
    """
          The test verifies the reachability of
          the following DFA and MDP combination
          MDP 1:
          with states (0,..,3) and no actions.

              [0] --> [1]--0.5->[2]   [3]
                      |--0.5---------->|
               |<----------------|

          DFA 1: with label {l=1,l=0}

               [0]-l=1->[[1]]

          Connection:

               {0: set([0]), 1: set([1]), 2: set([0]),3: set([0])}

          In *test_occupation_lp*, the following tests have been implemented:

          test_occupation1 tests MDP1 and DFA1:

          | test  | s0 = 0| s0 = 1| s0 = 2 |
          |-------|-------|-------|--------|
          | rprob | 0.5    | 1    |   0    |
        :return:
        """

    def setUp(self):

        self.network = POMDPNetwork()

        T0 = np.array([[0, 1, 0, 0],
                       [0, 0, 0.5, 0.5],
                       [1, 0, 0, 0],
                       [0, 0, 0, 1]]);

        self.network.add_pomdp(POMDP([T0], input_names=['a'], state_name='s'))

        T0 = np.array([[1, 0],
                       [0, 1]]);
        T1 = np.array([[0, 1],
                       [0, 1]]);
        self.network.add_pomdp(POMDP([T0, T1], input_names=['l'],
                                state_name='q'))

        self.network_nondet = copy.deepcopy(self.network)
        self.network.add_connection(['s'], 'l',
                               lambda s: {0: set([0]),
                                          1: set([1]),
                                          2: set([0]),
                                          3: set([0])}[s])
        self.network_nondet.add_connection(['s'], 'l',
                                    lambda s: {0: set([0]),
                                               1: set([0,1]),
                                               2: set([0]),
                                               3: set([0])}[s])

        self.mdp1 = diagonal(get_T_uxXz(self.network.pomdps['s']), 2, 3)
        self.dfa1 = diagonal(get_T_uxXz(self.network.pomdps['q']), 2, 3)

        accept = np.zeros((4,2))
        accept[:,1] = 1

        self.val_list, pol_list = solve_reach(self.network, accept)


    def test_solve_robust(self):

        # Define target set for value iteration
        conn = self.network.connections[0][2]

        #  test with s0=0
        reach_prob, _ = solve_robust(self.mdp1, self.dfa1, conn, s0=0, q0=0, q_target=1)
        np.testing.assert_almost_equal(reach_prob, self.val_list[0][0, 0])

        #  test with s0=1
        reach_prob, _ = solve_robust(self.mdp1, self.dfa1, conn, s0=1, q0=0, q_target=1)
        np.testing.assert_almost_equal(reach_prob, self.val_list[0][1, 0])

        #  test with s0=2
        reach_prob, _ = solve_robust(self.mdp1, self.dfa1, conn, s0=2, q0=0, q_target=1)
        np.testing.assert_almost_equal(reach_prob, self.val_list[0][2, 0])

    def test_solve_exact(self):
        conn = self.network.connections[0][2]

        reach_prob, _ = solve_exact(self.mdp1, self.dfa1, conn, s0=0, q0=0, q_target=1)
        np.testing.assert_almost_equal(reach_prob, self.val_list[0][0, 0])
        #  test with s0=1
        reach_prob, _ = solve_exact(self.mdp1, self.dfa1, conn, s0=1, q0=0, q_target=1)
        np.testing.assert_almost_equal(reach_prob, self.val_list[0][1, 0])
        #  test with s0=2
        reach_prob, _ = solve_exact(self.mdp1, self.dfa1, conn, s0=2, q0=0, q_target=1)
        np.testing.assert_almost_equal(reach_prob, self.val_list[0][2, 0])

    def test_occupation_lp_new(self):
        conn = self.network.connections[0][2]

        reach_prob, _ = occupation_lp_new(self.mdp1, self.dfa1, conn, s0=0, q0=0, q_target=1)
        np.testing.assert_almost_equal(reach_prob, self.val_list[0][0, 0])
        #  test with s0=1
        reach_prob, _ = occupation_lp_new(self.mdp1, self.dfa1, conn, s0=1, q0=0, q_target=1)
        np.testing.assert_almost_equal(reach_prob, self.val_list[0][1, 0])

        #  test with s0=2
        reach_prob, _ = occupation_lp_new(self.mdp1, self.dfa1, conn, s0=2, q0=0, q_target=1)
        np.testing.assert_almost_equal(reach_prob, self.val_list[0][2, 0])


    def test_solve_ltl(self):
        conn = self.network.connections[0][2]

        strat = np.array([[0]*(self.mdp1.shape[2]),[1]*(self.mdp1.shape[2])]).transpose()
        strat = strat[conn.transpose()]

        model,reach_prob = solve_ltl(self.mdp1, self.dfa1, strat,0, s0=0, q0=0, q_target=1)
        np.testing.assert_almost_equal(reach_prob['primal objective'], self.val_list[0][0, 0])
        #  test with s0=1
        model, reach_prob = solve_ltl(self.mdp1, self.dfa1, strat, 0, s0=2, q0=0, q_target=1)
        np.testing.assert_almost_equal(reach_prob['primal objective'], self.val_list[0][2, 0])
        #  test with s0=2
        model, reach_prob = solve_ltl(self.mdp1, self.dfa1, strat, 0, s0=1, q0=0, q_target=1)
        np.testing.assert_almost_equal(reach_prob['primal objective'], self.val_list[0][1, 0])



    def test_MDP1_and_DFA1_target1_nondet(self):
        """
                The test verifies the reachability of
                the following DFA and MDP combination
                MDP 1:
                with states (0,..,3) and no actions.

                    [0] --> [1]--0.5->[2]   [3]
                            |--0.5---------->|
                     |<----------------|

                DFA 1: with label {l=1,l=0}

                     [0]-l=1->[[1]]

                Connection:

                     {0: set([0]), 1: set([1]), 2: set([0]),3: set([0])}

                In *test_occupation_lp*, the following tests have been implemented:

                test_occupation1 tests MDP1 and DFA1:

                | test  | s0 = 0| s0 = 1| s0 = 2 |
                |-------|-------|-------|--------|
                | rprob | 0.5    | 1    |   0    |    """



        # Define target set
        accept = np.zeros((4,2))
        accept[:,1] = 1

        val_list, pol_list = solve_reach(self.network_nondet, accept)

        conn = self.network_nondet.connections[0][2]


        reach_prob, _ = occupation_lp_new(self.mdp1, self.dfa1, conn, s0=0, q0=0, q_target=1)
        np.testing.assert_almost_equal(reach_prob, val_list[0][0, 0])

        reach_prob, _ = occupation_lp_new(self.mdp1, self.dfa1, conn, s0=1, q0=0, q_target=1)
        np.testing.assert_almost_equal(reach_prob, val_list[0][1, 0])

        reach_prob, _ = occupation_lp_new(self.mdp1, self.dfa1, conn, s0=2, q0=0, q_target=1)
        np.testing.assert_almost_equal(reach_prob, val_list[0][2, 0])

    def test_MDP1(self):
        t = time.time()
        delta = 0.01
        model,sol = solve_pure_occ(self.mdp1.todense().tolist(), [len(self.mdp1[0])-1], 0, delta)
        print("Solution time", time.time()-t)
        print("Max occupancy state", np.argmax(sol['x']))
        print("Min occupancy value", min(sol['x']))


class BenchTests(unittest.TestCase):
    def setUp(self):
        from Demos.demo_Models import simple_robot
        import polytope as pc

        self.Robot = simple_robot()
        self.Robot.input_space = pc.box2poly(np.kron(
            np.ones((self.Robot.m, 1)), np.array([[-1, 1]])))  # continuous set of inputs
        self.Robot.state_space = pc.box2poly(np.kron(
            np.ones((self.Robot.dim, 1)), np.array([[-10, 10]])))  # X space

    def test_basic(self):
        # test lp that pushes robot to  sink state = leaving the gridded area

        print("-------Grid the robots state space------")
        #  set the state spaces:
        from Reduce.Gridding import grid as griddd


        d_opt = np.array([[0.69294], [0.721]])  # tuned gridding ratio from a previous paper
        d = 0.6 * d_opt  # with distance measure 0.6=default
        un = 3

        Ms, srep  = griddd(self.Robot, d, un=un)


        T00 = Ms.transition
        print(len(T00[0]))
        t = time.time()
        delta = 0.01
        print(np.max(np.sum(T00, axis=2)))
        model, sol = solve_pure_occ(T00.tolist(), len(T00[0])-1, 0, delta)

        print(time.time()-t)


    def test_demo(self):
        from Demos.demo_Models import simple_robot
        import polytope as pc
        import numpy as np


        print("-------Grid the robots state space------")
        #  set the state spaces:
        from Reduce.Gridding import grid as griddd

        d_opt = np.array([[0.69294], [0.721]])  # tuned gridding ratio from a previous paper
        d = 1.6 * d_opt  # with distance measure 0.6=default
        un = 3

        Ms, srep = griddd(self.Robot, d, un=un)

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
        Robotmdp = diagonal(get_T_uxXz(network.pomdps['s']), 2, 3)
        Robotdfa = diagonal(get_T_uxXz(network.pomdps['q']), 2, 3)
        conn = network.connections[0][2]
        delt = []
        reach = []
        val = []
        for delta in np.linspace(0.0001, 0.05, 5):
            t = time.time()
            delt += [delta]
            reach_prob, val2 = solve_delta(Robotmdp, Robotdfa, conn, delta, s0=0, q0=0, q_target=1)
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
            Robotmdp = diagonal(get_T_uxXz(network.pomdps['s']), 2, 3)
            Robotdfa = diagonal(get_T_uxXz(network.pomdps['q']), 2, 3)
            conn = network.connections[0][2].transpose()

            t=time.time()
            delta = 0.01
            delt += [delta]
            # reach_prob, val2 = solve_delta(self.mdp1, self.dfa1, conn, delta, s0=0, q0=0, q_target=1)
            strat = np.array([[0] * (Robotmdp.shape[2]), [1] * (Robotmdp.shape[2])]).transpose()
            strat = strat[conn]

            model, reach_prob = solve_ltl(Robotmdp.todense(), Robotdfa.todense(), strat, delta, s0=0, q0=0, q_target=1)
            # np.testing.assert_almost_equal(reach_prob['primal objective'], val_list[0][0, 0], decimal=5)

            reach += [reach_prob['primal objective']]
            t_list += [time.time()-t]
            print(time.time()-t)

            # print(val2)
        # np.testing.assert_almost_equal(reach_prob, val_list[0][0, 0],decimal=5)
        print("reach prob",reach)
        print("solve time ",t_list)



    if __name__ == "__main__":
        unittest.main()
