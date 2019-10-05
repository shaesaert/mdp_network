from itertools import product

import scipy.sparse as sp

from best.models.pomdp import POMDP, POMDPNetwork
from best.solvers.valiter import solve_reach
from best.solvers.occupation_lp import *
from best.models.pomdp_sparse_utils import get_T_uxX, diagonal
from best.solvers.optimization_wrappers import Constraint, solve_ilp

import numpy as np

def occupation_lp_new(P_asS, P_lqQ, conn_mat, s0, q0, q_target, delta=0):

  P_asS = P_asS.todense()
  P_lqQ = P_lqQ.todense()

  na = P_asS.shape[0]
  ns = P_asS.shape[1]
  nl = P_lqQ.shape[0]
  nq = P_lqQ.shape[1]
  nk = 2*nl + 2

  T = [q_target]                     # target set
  F = list(set(range(nq)) - set(T))  # not target set

  nF = len(F)
  nT = len(T)

  num_varP = 1
  num_varX = na * ns * nF
  num_varD1 = 0
  num_varD2 = 0

  A_iq = sp.coo_matrix((0, num_varP + num_varX + num_varD1))
  b_iq = np.zeros(0)

  A_eq = sp.coo_matrix((0, num_varP + num_varX + num_varD1))
  b_eq = np.zeros(0)

  sDet = set(s for s in range(ns) if sum(conn_mat[:, s]) == 1)  # deterministic transitions
  sNDet = set(range(ns)) - sDet  # nondeterministic transitions

  if False:    # use non-deterministic formulation for all, to compare with solve_robust
    sDet = set()
    sNDet = set(s for s in range(ns))

  poly_A = np.vstack([np.eye(nl), -np.eye(nl), np.ones([1, nl]), -np.ones([1, nl])])

  # VARIABLES [P   x          dummies]
  # OF SIZE   [1   na*ns*nq   nk * D ]

  ##########################################
  ########### FIRST INEQUALITY #############
  ##########################################

  for F_it, S_it in product(range(nF), range(ns)):
    Q_it = F[F_it]

    if S_it in sDet:
      # deterministic
      l_it = np.nonzero(conn_mat[:, S_it])[0]

      new_row = np.zeros((1, num_varP + num_varX + num_varD1))

      # add sum over a
      new_row_suma_idx = np.ravel_multi_index((np.arange(na, dtype=np.uint32), np.ones(na, dtype=np.uint32) * S_it, np.ones(na, dtype=np.uint32) * F_it), (na, ns, nF))
      new_row[0, num_varP + new_row_suma_idx] = 1
      
      # subtract sum over sqa
      for (a_it, s_it, f_it) in product(range(na), range(ns), range(nF)):
        q_it = F[f_it]
        new_row[0, num_varP + np.ravel_multi_index((a_it, s_it, f_it), (na, ns, nF))] -= P_asS[a_it, s_it, S_it] * P_lqQ[l_it, q_it, Q_it]

      A_iq = sp.bmat([[A_iq], [sp.coo_matrix(new_row)]])
      b_iq = np.hstack([b_iq, 1 if S_it == s0 and Q_it == q0 else 0])

    else:
      # non-deterministic, add more dummy variables
      num_varD1 += nk
      A_iq = sp.bmat([[A_iq, sp.coo_matrix((A_iq.shape[0], nk))]])
      A_eq = sp.bmat([[A_eq, sp.coo_matrix((A_eq.shape[0], nk))]])

      # INEQUALITY
      new_row = np.zeros((1, num_varP + num_varX + num_varD1))

      # sum over a
      new_row_suma_idx = np.ravel_multi_index((np.arange(na, dtype=np.uint32), np.ones(na, dtype=np.uint32) * S_it, np.ones(na, dtype=np.uint32) * F_it), (na, ns, nF))
      new_row[0, num_varP + new_row_suma_idx] = 1

      # polytope containing possible labels
      new_row[0, -nk:] = np.hstack([conn_mat[:, S_it], np.zeros(nl), 1, -1])

      A_iq = sp.bmat([[A_iq], [sp.coo_matrix(new_row)]])
      b_iq = np.hstack([b_iq, 1 if S_it == s0 and Q_it == q0 else 0])

      # EQUALITY
      
      # sum over asq
      new_block = np.zeros((nl, num_varP + num_varX + num_varD1))
      for l_it in range(nl):
        new_block[l_it, -nk:] = poly_A[:, l_it]
        for (a_it, s_it, f_it) in product(range(na), range(ns), range(nF)):
          q_it = F[f_it]
          new_block[l_it, num_varP + np.ravel_multi_index((a_it, s_it, f_it), (na, ns, nF))] += P_asS[a_it, s_it, S_it] * P_lqQ[l_it, q_it, Q_it]
        

      A_eq = sp.bmat([[A_eq], [sp.coo_matrix(new_block)]])
      b_eq = np.hstack([b_eq, np.zeros(nl)])

  ##########################################
  ########### SECOND INEQUALITY ############
  ##########################################

  num_varD2 = nk * len(sNDet)

  A_iq = sp.bmat([[A_iq, sp.coo_matrix((A_iq.shape[0], num_varD2))]])
  A_eq = sp.bmat([[A_eq, sp.coo_matrix((A_eq.shape[0], num_varD2))]])

  # INEQUALITY
  new_row = np.zeros((1, num_varP + num_varX + num_varD1 + num_varD2))
  new_row[0, 0] = 1  # coefficient in front of P

  # sum over all deterministic s'
  for (a_it, s_it, S_it, f_it, t_it) in product(range(na), range(ns), sDet, range(nF), range(nT)):
    q_it = F[f_it]
    Q_it = T[t_it]
    l_it = np.nonzero(conn_mat[:, S_it])[0]
    new_row[0, 1 + np.ravel_multi_index((a_it, s_it, f_it), (na, ns, nF))] -= P_asS[a_it, s_it, S_it] * P_lqQ[l_it, q_it, Q_it]

  # sum over all nondeterministic s'
  for (sn_it, s_it) in enumerate(sNDet):
    b_k = np.hstack([conn_mat[:, s_it], np.zeros(nl), 1, -1])
    for k_it in range(nk):
      sk_idx = np.ravel_multi_index((sn_it, k_it), (len(sNDet), nk))
      new_row[0, num_varP + num_varX + num_varD1 + sk_idx] += b_k[k_it]

  A_iq = sp.bmat([[A_iq], [sp.coo_matrix(new_row)]])
  b_iq = np.hstack([b_iq, 0])

  # EQUALITY
  new_block = np.zeros((nl * len(sNDet), num_varP + num_varX + num_varD1 + num_varD2))

  # sum over all non-determninistc s
  for (Sn_it, S_it), l_it in product(enumerate(sNDet), range(nl)):

    Sl_idx = np.ravel_multi_index((Sn_it, l_it), (len(sNDet), nl))

    # sum over saqQ
    for (a_it, s_it, f_it, t_it) in product(range(na), range(ns), range(nF), range(nT)):
      q_it = F[f_it]
      Q_it = T[t_it]
      x_idx = np.ravel_multi_index((a_it, s_it, q_it), (na, ns, nF))
      new_block[Sl_idx, num_varP + x_idx] += P_asS[a_it, s_it, S_it] * P_lqQ[l_it, q_it, Q_it]

    # sum over sk
    for k_it in range(nk):
      sk_idx = np.ravel_multi_index((Sn_it, k_it), (len(sNDet), nk))
      new_block[Sl_idx, num_varP + num_varX + num_varD1 + sk_idx] += poly_A[k_it, l_it]

  A_eq = sp.bmat([[A_eq], [sp.coo_matrix(new_block)]])
  b_eq = np.hstack([b_eq, np.zeros(num_varD2)])


  constr = Constraint(A_eq=A_eq, b_eq=b_eq, A_iq=A_iq, b_iq=b_iq)

  objective = np.hstack([-1, np.ones(num_varX)*delta, np.zeros(num_varD1 + num_varD2)])

  sol = solve_ilp(objective, constr, J_int=[])

  if sol['status'] == 'optimal':
    return -sol['primal objective'], sol['x'][num_varP: num_varP+num_varX].reshape((na, ns, nF))
  else:
    print("solver returned {}".format(sol['status']))
    return -1, -1, -1
