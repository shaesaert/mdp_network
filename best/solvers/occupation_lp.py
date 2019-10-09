'''Methods for solving reachability problems via occupation measure LPs

Currently only support two-part networks (e.g., MDP + automaton) with potentially
nondeterministic connections
'''
import numpy as np
import scipy.sparse as sp
import sparse
import time
import warnings
from best.models.pomdp_sparse_utils import diagonal, diag

try:
    import gurobipy as grb

    TIME_LIMIT = 10 * 3600

except Exception as e:
    print("warning: gurobi not found")

from best.solvers.optimization_wrappers import Constraint, solve_ilp



def solve_delta(P_asS, P_lqQ, conn_mat, delta, s0, q0, q_target):
    """  Solve reachability with delta error in transition probabilities
    :param P_asS: TRansition matrix of systems
    :param P_lqQ: Transition matrix of DFA
    :param conn_mat:  Connection matrix that links labels to enabled transitions in P_lq
    :param s0: Initial system state
    :param q0: Initial DFA state
    :param q_target:  accepting q_state
    :param delta: delta error for each transition
    :return:
    """
    t = time.time()
    na = P_asS.shape[0]  # number of actions of MDP
    ns = P_asS.shape[1]  # number of states of MDP
    nl = P_lqQ.shape[0]  # number of labels of DFA
    nq = P_lqQ.shape[1]  # number of states of DFA
    nk = nl*2+2

    if q_target != nq-1:
        raise Exception("case q_target not equal to highest q not implemented")

    idx_notarget = [i for i in range(nq) if i != q_target]
    idx_target = [q_target]
    P_lqQ_notarget = P_lqQ[:, :, idx_notarget]
    P_lqQ_notarget = P_lqQ_notarget[:, idx_notarget, :]  # transitions from Q \ T to Q \ T
    P_lqQ_sep = P_lqQ[:, idx_notarget, :]
    P_lqQ_sep = P_lqQ_sep[:, :, idx_target]              # transitions from Q \ T to T

    nq_notarget = nq - len(idx_target)

    num_varP = 1                      # objective (reach probability)
    num_varX = na * ns * nq_notarget  # x variable (occupation measures)

    P_lS = sparse.COO(conn_mat)

    ##################
    # Constraint 12b #
    ##################

    # right-hand side
    sum_a = sparse.COO([range(na)], np.ones(na))
    R_aSsQq = sparse.tensordot(sum_a, sparse.tensordot(sparse.eye(ns), sparse.eye(nq_notarget), axes=-1), axes=-1) #
    R_b_QS_asq = R_aSsQq.transpose([3, 1, 0, 2, 4]).reshape([ns * nq_notarget, na * ns * nq_notarget])

    # left-hand side
    L_SlQasq = sparse.tensordot(P_asS, P_lqQ_notarget, axes=-1).transpose([2, 3, 5, 0, 1, 4]) #
    L_SQasqS = sparse.tensordot(L_SlQasq, P_lS, axes=[[1], [0]])
    L_QSasq = diagonal(L_SQasqS, axis1=0, axis2=5).transpose([0,4,1,2,3])
    L_QS_asq_sp = L_QSasq.reshape([nq_notarget * ns, na * ns * nq_notarget]).to_scipy_sparse()

    # TODO: indexing needs fix to have q_target not being the last one
    b_iq_b = np.zeros(ns * nq_notarget)
    b_iq_b[np.ravel_multi_index((s0, q0), (ns, nq_notarget))] = 1.

    c = Constraint(A_iq=sp.bmat([[sp.coo_matrix((ns * nq_notarget, num_varP)),
                                  R_b_QS_asq.to_scipy_sparse() - L_QS_asq_sp]]),
                   b_iq=b_iq_b)

    ##################
    # Constraint 12c #
    ##################

    # right-hand side
    R_e_asSlqQ = sparse.tensordot(P_asS, P_lqQ_sep, axes=-1)     #
    R_e_Slasq = R_e_asSlqQ.sum(axis=[5]).transpose([2,3,0,1,4])  #
    R_asq = sparse.tensordot(R_e_Slasq, P_lS, axes=[[1, 0], [0, 1]])
    R_asq_sp = R_asq.reshape([1, na*ns*nq_notarget]).to_scipy_sparse()

    c &= Constraint(A_iq=sp.bmat([[1, -R_asq_sp]]), b_iq=[0])

    ##################
    #### Solve it ####
    ##################

    objective = np.ones(num_varP + num_varX)*delta
    objective[0] = -1  # maximize P
    t_init = time.time()
    print(["Initiate solver: ", t_init- t] )
    sol = solve_ilp(objective, c, J_int=[])
    t_solve = time.time()
    print(["Run solver: ", t_solve-t_init])

    if sol['status'] == 'optimal':
        return -sol['primal objective'], sol['x'][num_varP: num_varP+num_varX].reshape((na, ns, nq_notarget))
    else:
        print("solver returned {}".format(sol['status']))
        return -1, -1


def solve_robust(P_asS, P_lqQ, conn_mat, s0, q0, q_target):
    # formulate and solve robust LP for system
    # consisting of two mdps P_{uxx'} and Q_{vyy'},
    # where x' yields (nondeterministic) inputs v

    t = time.time()
    na = P_asS.shape[0]
    ns = P_asS.shape[1]
    nl = P_lqQ.shape[0]
    nq = P_lqQ.shape[1]
    nk = nl * 2 + 2

    if q_target != nq - 1:
        raise Exception("case q_target not equal to highest q not implemented")

    idx_notarget = [i for i in range(nq) if i != q_target]
    idx_target = [q_target]
    P_lqQ_notarget = P_lqQ[:, :, idx_notarget]
    P_lqQ_notarget = P_lqQ_notarget[:, idx_notarget, :]  # transitions from Q \ T to Q \ T
    P_lqQ_sep = P_lqQ[:, idx_notarget, :]
    P_lqQ_sep = P_lqQ_sep[:, :, idx_target]  # transitions from Q \ T to T

    nq_notarget = nq - len(idx_target)

    num_varP = 1  # objective (reach probability)
    num_varX = na * ns * nq_notarget  # x variable (occupation measures)
    num_var1 = nk * ns * nq_notarget  # dummy variable 1
    num_var2 = nk * ns  # dummy variable 2

    # Extract uncertainty polyhedron
    A_poly_Skl = sparse.stack(
        [sparse.COO(np.vstack([np.eye(nl), -np.eye(nl), np.ones([1, nl]), -np.ones([1, nl])])) for state in range(ns)])
    b_poly_Sk = sparse.stack([sparse.COO(np.hstack([conn_mat[:, state], np.zeros(nl), 1, -1])) for state in range(ns)])

    ##################
    # Constraint 15b #
    ##################
    num_iq_b = ns * nq_notarget
    # Left-hand side
    L_SkS = diag(b_poly_Sk, axis=0)
    L_SkSQQ = sparse.tensordot(L_SkS, sparse.eye(nq_notarget), axes=-1)
    L_b_QS_QSk = L_SkSQQ.transpose([3, 0, 4, 2, 1]).reshape([nq_notarget * ns, nq_notarget * ns * nk])  # <--
    # Right-hand side
    sum_a = sparse.COO([range(na)], np.ones(na))
    R_aSsQq = sparse.tensordot(sum_a, sparse.tensordot(sparse.eye(ns), sparse.eye(nq_notarget), axes=-1), axes=-1)
    R_b_QS_asq = R_aSsQq.transpose([3, 1, 0, 2, 4]).reshape([ns * nq_notarget, na * ns * nq_notarget])

    A_iq_b = sp.bmat([[sp.coo_matrix((num_iq_b, num_varP)),  # P
                       R_b_QS_asq.to_scipy_sparse(),  # X
                       L_b_QS_QSk.to_scipy_sparse(),  # V1
                       sp.coo_matrix((num_iq_b, num_var2))]])  # V2
    # TODO: indexing needs fix to have q_target not being the last one
    b_iq_b = np.zeros(num_iq_b)
    b_iq_b[np.ravel_multi_index((s0, q0), (ns, nq_notarget))] = 1.
    c = Constraint(A_iq=A_iq_b, b_iq=b_iq_b)

    ##################
    # Constraint 15c #
    ##################
    num_eq_c = ns * nl * nq_notarget
    # Left-hand side
    L_SklS = diag(A_poly_Skl, axis=0)
    L_SklSQQ = sparse.tensordot(L_SklS, sparse.eye(nq_notarget), axes=-1)
    L_c_SlQ_QSk = L_SklSQQ.transpose([0, 2, 4, 5, 3, 1]).reshape([ns * nl * nq_notarget, nq_notarget * ns * nk])
    # Right-hand side
    R_SlQasq_c = sparse.tensordot(P_asS, P_lqQ_notarget, axes=-1).transpose([2, 3, 5, 0, 1, 4])
    R_c_SlQ_asq = R_SlQasq_c.reshape([ns * nl * nq_notarget, na * ns * nq_notarget])

    A_eq_c = sp.bmat([[sp.coo_matrix((num_eq_c, num_varP)),  # P
                       R_c_SlQ_asq.to_scipy_sparse(),  # X
                       L_c_SlQ_QSk.to_scipy_sparse(),  # V1
                       sp.coo_matrix((num_eq_c, num_var2))]])  # V2
    c &= Constraint(A_eq=A_eq_c, b_eq=np.zeros(num_eq_c))

    ##################
    # Constraint 15d #
    ##################
    num_iq_d = 1
    L_d_sk = b_poly_Sk.reshape((1, ns * nk))
    A_iq_d = sp.bmat([[1,  # P
                       sp.coo_matrix((num_iq_d, num_varX)),  # X
                       sp.coo_matrix((num_iq_d, num_var1)),  # V1
                       L_d_sk.to_scipy_sparse()]])  # V2
    c &= Constraint(A_iq=A_iq_d, b_iq=np.zeros(num_iq_d))

    ##################
    # Constraint 15e #
    ##################
    num_eq_e = nl * ns;
    # Left-hand side
    L_e_Sl_sk = diag(A_poly_Skl, axis=0).transpose([3, 2, 0, 1]).reshape([ns * nl, ns * nk])
    # Right-hand side
    R_e_asSlqQ = sparse.tensordot(P_asS, P_lqQ_sep, axes=-1)
    R_e_Slasq = R_e_asSlqQ.sum(axis=[5]).transpose([2, 3, 0, 1, 4])
    R_e_Sl_asq_e = R_e_Slasq.reshape([ns * nl, na * ns * nq_notarget])

    A_eq_e = sp.bmat([[sp.coo_matrix((num_eq_e, num_varP)),  # P
                       R_e_Sl_asq_e.to_scipy_sparse(),  # X
                       sp.coo_matrix((num_eq_e, num_var1)),  # V1
                       L_e_Sl_sk.to_scipy_sparse()]])  # V2
    c &= Constraint(A_eq=A_eq_e, b_eq=np.zeros(num_eq_e))

    ##################
    #### Solve it ####
    ##################

    objective = np.zeros(num_varP + num_varX + num_var1 + num_var2)
    objective[0] = -1  # maximize P
    t_init = time.time()
    print(["Initiate solver: ", t_init - t])
    sol = solve_ilp(objective, c, J_int=[])
    t_solve = time.time()
    print(["Run solver: ", t_solve - t_init])

    ########################################################

    if sol['status'] == 'optimal':
        return sol['x'][0], sol['x'][num_varP: num_varP + num_varX].reshape((na, ns, nq_notarget))
    else:
        print("solver returned {}".format(sol['status']))
        return -1, -1


def solve_ltl(P_asS, P_lqQ, strat, delta, s0, q0, q_target):
    """

    :param P_asS:  TRansition matrix of the MDP
    :param P_lqQ: Transition matrix of the DFA
    :param strat: Strategy S-> label (this has to be deterministic for this function)
    :param delta: delta error in the transition probabilities
    :param s0: initial state in the MDP
    :param q0: initial q in the DFA
    :param q_target: target q in the DFA
    :return: model (gurobi),  sol
    """
     # solve pure reach problem for stochastic system.
    # T is list finite states. For this alg. to work T needs to be quite small.
    if not isinstance(q_target, list): q_target = [q_target]
    t = time.time()
    nS = P_asS.shape[1]  # number of states not in T
    na = P_asS.shape[0]  # number of actions

    nQ = len(P_lqQ[0])-len(q_target)  # number of states in Sq\q_target
    nl = P_lqQ.shape[0]  # number of labels of DFA
    nT = len(q_target)
    QT = [q for q in range(nQ + nT) if q not in q_target] # Q not in T

    model = grb.Model("prob0")
    x_qsa = dict()  # state action occupancy
    print("create empty model", time.time()-t)
    for q in QT:
        # figure out whether there exists a transition from q to q_target

        checkprob = False
        if max(max(P_lqQ[:, q, q_target])) != 0:
            checkprob = True # there exists a transition to q_target
            assert (max(max(P_lqQ[:, q, q_target])) == 1)
        p_list = []
        for a in range(na):
            p_list = []
            for s in range(nS):
                if checkprob:

                    prob = sum([P_asS[a,s,sn] for qn in q_target
                          for sn in range(nS) if ( (P_lqQ[strat[sn], q, qn] == 1))])
                else:
                    prob = 0
                x_qsa[q, s, a] = model.addVar(vtype=grb.GRB.CONTINUOUS,
                                              obj=-delta+prob)

    print('added variables', time.time()-t)

    for qn in QT:
        for sn in range(nS):  # compute occupation

            rhs = 1 if (sn == s0) & (qn == q0) else 0

            model.addConstr(grb.quicksum(x_qsa[qn, sn, a] for a in range(na))
                            - grb.quicksum(x_qsa[q,s, a] * P_asS[a,s,sn] for q in QT for a in range(na)
                            for s in range(nS) if ((P_lqQ[strat[sn], q, qn]==1))) <= rhs )
    print('added constraints', time.time()-t)


    model.ModelSense = -1
    model.update()
    # model.optimize()

    # Attempt to set an initial feasible solution (in this case to an optimal solution)
    for q in QT:
        for s in range(nS):
            for a in range(na):
                x_qsa[q, s, a].start = 0

    model.optimize()


    sol = {}
    if model.status == grb.GRB.status.OPTIMAL:
        sol['x'] = np.array([var.x for var in x_qsa.values()])
        sol['primal objective'] = model.objVal
    if model.status in [2, 3, 5]:
        sol['rcode'] = model.status
    else:
        sol['rcode'] = 1

    return model, sol




def solve_exact(P_asS, P_lqQ, conn_mat, s0, q0, q_target):
    """
    Outdated version of the LP occupation measure computation of reachability.
     This version is obsolete and can be replaced by the solve_delta"

    :param P_asS: TRansition matrix of systems
    :param P_lqQ: Transition matrix of DFA
    :param conn_mat:  Connection matrix that links labels to enabled transitions in P_lq
    :param s0: Initial system state
    :param q0: Initial DFA state
    :param q_target:  accepting q_state
    :return: 
    """
    warnings.warn('solve_exact is obsolete and should not be used anymore, '
                  'it can be replaced by the faster solve_delta with delta=0. ')
    na = P_asS.shape[0]
    ns = P_asS.shape[1]
    nl = P_lqQ.shape[0]
    nq = P_lqQ.shape[1]
    nk = nl * 2 + 2

    if q_target != nq - 1:
        raise Exception("case q_target not equal to highest q not implemented")

    idx_notarget = [i for i in range(nq) if i != q_target]
    idx_target = [q_target]
    P_lqQ_notarget = P_lqQ[:, :, idx_notarget]
    P_lqQ_notarget = P_lqQ_notarget[:, idx_notarget, :]  # transitions from Q \ T to Q \ T
    P_lqQ_sep = P_lqQ[:, idx_notarget, :]
    P_lqQ_sep = P_lqQ_sep[:, :, idx_target]  # transitions from Q \ T to T

    nq_notarget = nq - len(idx_target)

    num_varP = 1  # objective (reach probability)
    num_varX = na * ns * nq_notarget  # x variable (occupation measures)

    P_lS = sparse.COO(conn_mat)

    ##################
    # Constraint 12b #
    ##################

    # right-hand side
    sum_a = sparse.COO([range(na)], np.ones(na))
    R_aSsQq = sparse.tensordot(sum_a, sparse.tensordot(sparse.eye(ns), sparse.eye(nq_notarget), axes=-1), axes=-1)  #
    R_b_QS_asq = R_aSsQq.transpose([3, 1, 0, 2, 4]).reshape([ns * nq_notarget, na * ns * nq_notarget])

    # left-hand side
    L_SlQasq = sparse.tensordot(P_asS, P_lqQ_notarget, axes=-1).transpose([2, 3, 5, 0, 1, 4])  #
    L_SQasqS = sparse.tensordot(L_SlQasq, P_lS, axes=[[1], [0]])
    L_QSasq = diagonal(L_SQasqS, axis1=0, axis2=5).transpose([0, 4, 1, 2, 3])
    L_QS_asq_sp = L_QSasq.reshape([nq_notarget * ns, na * ns * nq_notarget]).to_scipy_sparse()

    # TODO: indexing needs fix to have q_target not being the last one
    b_iq_b = np.zeros(ns * nq_notarget)
    b_iq_b[np.ravel_multi_index((s0, q0), (ns, nq_notarget))] = 1.

    c = Constraint(A_iq=sp.bmat([[sp.coo_matrix((ns * nq_notarget, num_varP)),
                                  R_b_QS_asq.to_scipy_sparse() - L_QS_asq_sp]]),
                   b_iq=b_iq_b)

    ##################
    # Constraint 12c #
    ##################

    # right-hand side
    R_e_asSlqQ = sparse.tensordot(P_asS, P_lqQ_sep, axes=-1)  #
    R_e_Slasq = R_e_asSlqQ.sum(axis=[5]).transpose([2, 3, 0, 1, 4])  #
    R_asq = sparse.tensordot(R_e_Slasq, P_lS, axes=[[1, 0], [0, 1]])
    R_asq_sp = R_asq.reshape([1, na * ns * nq_notarget]).to_scipy_sparse()

    c &= Constraint(A_iq=sp.bmat([[1, -R_asq_sp]]), b_iq=[0])

    ##################
    #### Solve it ####
    ##################

    objective = np.zeros(num_varP + num_varX)
    objective[0] = -1  # maximize P

    sol = solve_ilp(objective, c, J_int=[])

    if sol['status'] == 'optimal':
        return sol['x'][0], sol['x'][num_varP: num_varP + num_varX].reshape((na, ns, nq_notarget))
    else:
        print("solver returned {}".format(sol['status']))
        return -1, -1



def solve_pure_occ(P, T, S0, delta):
    # solve pure reach problem for stochastic system. T is list finite states. For this alg. to work T needs to be quite small.
    if not isinstance(T, list): T = [T]

    nST = len(P[0]) - len(T)  # number of states not in T
    nT = len(T)  # number of states in T
    na = len(P)  # number of actions


    ST = [s for s in range(nST + nT) if s not in T]


    model = grb.Model("prob0")
    x_sa = dict()  # state action occupancy
    for a in range(na):
        for s in ST:
            x_sa[s, a] = model.addVar(vtype=grb.GRB.CONTINUOUS,
                                      obj=-delta + sum(P[a][s][t]
                                                       for t in T if (P[a][s][t] > 10e-10)))
    model.update()


    for sn in ST:  # compute occupation
        rhs = 1 if (sn == S0) else 0
        model.addConstr(grb.quicksum(x_sa[sn, a] for a in range(na))
                 - grb.quicksum(x_sa[s, a] * P[a][s][sn] for a in range(na)
                                for s in ST if (P[a][s][sn] > 10e-10)) <= rhs)

    model.update()
    model.ModelSense = -1

    model.optimize()

    sol = {}
    if model.status == grb.GRB.status.OPTIMAL:
        sol['x'] = np.array([var.x for var in x_sa.values()])
        sol['primal objective'] = model.objVal
    if model.status in [2, 3, 5]:
        sol['rcode'] = model.status
    else:
        sol['rcode'] = 1

    return model, sol

