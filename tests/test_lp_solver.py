#!/usr/bin/env python
"""Short title.

Long explanation
"""


import unittest

import gurobipy as grb
import best.solvers.optimization_wrappers

from best.solvers.occupation_lp import *


class test_lp_solvers(unittest.TestCase):
    """Demo test case 2."""

    def test_lp(self):
        b = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
        c = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)

        # Create a new model
        model = grb.Model("lp0")

        y = model.addVars(5,vtype=grb.GRB.CONTINUOUS,name = 'x')

        for i in range(5):
            model.addConstr(y[i] <= b[i])

        print(y)
        model.ModelSense = grb.GRB.MAXIMIZE
        obj = grb.quicksum(y[i] for i in range(5))

        model.setObjective(obj, grb.GRB.MAXIMIZE)

        print(model)
        model.optimize()

    def test_lp_prob(self):

        nST = 2  # number of states not in T
        nT = 1  # number of states in T
        na = 2  # number of actions
        P = [[[1,0,0], [0,1,0], [0,0,1]], [[.5,.2,.3], [.4,.6,0], [0,0,1]]]
        T = [2]  # the target state is the last state :)
        S0 = 0  # the initial state
        delta = 0.01

        # check the shape of P
        # should be 2 actions and 3 states
        print(len(P))

        model = grb.Model("prob0")
        y = model.addVars(nST,na,vtype=grb.GRB.CONTINUOUS,name = 'x')
        s = model.addVars(nST,vtype=grb.GRB.CONTINUOUS,name = 'slack')

        x = []
        for s in range(nST): # compute occupation
            x += [grb.quicksum(y[s,a] for a in range(na))]

        xi = []
        for sn in range(nST): # compute incoming occup
            # from s to s' sum_a x_sa Ps,a,s'
            print(sn)
            xi += [grb.quicksum(y[s,a]*P[a][s][sn] for a in range(na) for s in range(nST))]

        lhs = [x[i]-xi[i] for i in range(len(xi))]
        rhs = [0] * nST
        rhs[S0] = 1

        H = grb.quicksum(delta*y[s, a] for a in range(na) for s in range(nST))
        obj = grb.quicksum(y[s, a]*P[a][s][sn] for a in range(na) for s in range(nST) for sn in T) - H
        print('obj',obj)
        # b = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
        # c = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
        #
        #
        # # Create a new model
        #
        #
        for i in range(nST):
            model.addConstr(lhs[i] <= rhs[i])

        model.setObjective(obj, grb.GRB.MAXIMIZE)
        model.optimize()
        print(y)

    def test_lp_prob_v2(self):

        nST = 2  # number of states not in T
        nT = 1  # number of states in T
        na = 2  # number of actions
        P = [[[1,0,0], [0,1,0], [0,0,1]], [[.5,.2,.3], [.4,.6,0], [0,0,1]]]
        T = [2]  # the target state is the last state :)
        S0 = 0  # the initial state
        delta = 0.01

        # check the shape of P
        # should be 2 actions and 3 states
        print(len(P))

        model = grb.Model("prob0")
        y = dict()
        for a in range(na):
            for s in range(nST):
                y[s,a] = model.addVar( vtype=grb.GRB.CONTINUOUS,  obj=-delta+P[a][s][T[0]])
        model.update()

        x = []
        for s in range(nST): # compute occupation
            x += [grb.quicksum(y[s,a] for a in range(na))]

        xi = []
        for sn in range(nST): # compute incoming occup
            # from s to s' sum_a x_sa Ps,a,s'
            print(sn)
            xi += [grb.quicksum(y[s,a]*P[a][s][sn] for a in range(na) for s in range(nST))]

        lhs = [x[i]-xi[i] for i in range(len(xi))]
        rhs = [0] * nST
        rhs[S0] = 1

        # H = grb.quicksum(delta*y[s, a] for a in range(na) for s in range(nST))
        # obj = grb.quicksum(y[s, a]*P[a][s][sn] for a in range(na) for s in range(nST) for sn in T) - H
        # print('obj',obj)
        # b = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
        # c = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
        #
        #
        # # Create a new model
        #
        #
        for i in range(nST):
            model.addConstr(lhs[i] <= rhs[i])
            print(lhs[i],rhs[i])

        model.ModelSense = -1
        model.update()
        print('obj',model.getObjective())

        print('cnstrs 1', model.getConstrs()[0])
        print('cnstrs 2')
        print(model.getConstrs()[1])
        # model.setObjective(obj, grb.GRB.MAXIMIZE)
        print(model)
        model.optimize()
        print(y)


    def test_func(self):
        P = np.array( [[[1,   0,  0], [0,   1, 0], [0, 0, 1]],
                       [[.5, .2, .3], [.4, .6, 0], [0, 0, 1]]] )
        T = 2  # the target state is the last state :)
        S0 = 0  # the initial state
        delta = 0.01
        #(c: object, Aiq: object, biq: object, Aeq: object, beq: object, J_int: object, J_bin: object, output: object) -> object:

        Aiq = P[:,0:-1,0:-1]
        c = P[:,0:-1, -1]




        # sol = best.solvers.optimization_wrappers._solve_gurobi(P,T,S0,delta)
        # print(sol)

if __name__ == "__main__":
    unittest.main()