from itertools import product

import time
import scipy.sparse as sp

from best.models.pomdp import POMDP, POMDPNetwork
from best.solvers.valiter import solve_reach
from best.solvers.occupation_lp import *
from best.models.pomdp_sparse_utils import get_T_uxX, diagonal
from best.solvers.optimization_wrappers import Constraint, solve_ilp

import numpy as np

def occupation_lp_new(P_asS, P_lqQ, conn_mat, s0, q0, q_target):
  t = time.time()

  P_asS = P_asS.todense()
  P_lqQ = P_lqQ.todense()

  na = P_asS.shape[0]
  ns = P_asS.shape[1]
  nl = P_lqQ.shape[0]
  nq = P_lqQ.shape[1]
  nk = 2*nl + 2

  T = [q_target]
  F = list(set(range(nq)) - set(T))

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

  # VARIABLES [P   x          dummies]
  # OF SIZE   [1   na*ns*nq   nk * D ]

  ##########################################
  ########### FIRST INEQUALITY #############
  ##########################################

  for S_it, F_it in product(range(ns), range(nF)):
    Q_it = F[F_it]

    if S_it in sDet:
      # deterministic
      l_it = np.nonzero(conn_mat[:, S_it])[0]

      new_row_val = np.zeros((1, num_varP + num_varX + num_varD1))
      
      # sum over sqa
      for (a_it, s_it, f_it) in product(range(na), range(ns), range(nF)):
        q_it = F[f_it]
        new_row_val[0, num_varP+np.ravel_multi_index((a_it, s_it, f_it), (na, ns, nF))] -= P_asS[a_it, s_it, S_it] * P_lqQ[l_it, q_it, Q_it]

      # sum over a
      new_row_suma_idx = np.ravel_multi_index((np.arange(na, dtype=np.uint32), np.ones(na, dtype=np.uint32) * S_it, np.ones(na, dtype=np.uint32) * f_it), (na, ns, nF))
      new_row_val[0, num_varP + new_row_suma_idx] += 1
      A_iq = sp.bmat([[A_iq], [sp.coo_matrix(new_row_val)]])
      b_iq = np.hstack([b_iq, 1 if S_it == s0 and Q_it == q0 else 0])

    else:
      # non-deterministic, add more dummy variables
      num_varD1 += nk
      A_iq = sp.bmat([[A_iq, sp.coo_matrix((A_iq.shape[0], nk))]])
      A_eq = sp.bmat([[A_eq, sp.coo_matrix((A_eq.shape[0], nk))]])

      # polytope containing possible labels
      A_kl = np.vstack([np.eye(nl), -np.eye(nl), np.ones([1, nl]), -np.ones([1, nl])])
      b_k = np.hstack([conn_mat[:, S_it], np.zeros(nl), 1, -1])

      # INEQUALITY
      new_row_val = np.zeros((1, num_varP + num_varX + num_varD1))

      # sum over a
      new_row_suma_idx = np.ravel_multi_index((np.arange(na, dtype=np.uint32), np.ones(na, dtype=np.uint32) * S_it, np.ones(na, dtype=np.uint32) * Q_it), (na, ns, nq))
      new_row_val[0, num_varP + new_row_suma_idx] = 1

      # sum over k
      new_row_val[0, -nk:] = b_k

      A_iq = sp.bmat([[A_iq], [sp.coo_matrix(new_row_val)]])
      b_iq = np.hstack([b_iq, 1 if S_it == s0 and Q_it == q0 else 0])

      # EQUALITY
      
      # sum over sqa
      new_row_val = np.zeros((nl, num_varP + num_varX + num_varD1))
      for l_it in range(nl):
        for (a_it, s_it, f_it) in product(range(na), range(ns), range(nF)):
          q_it = F[f_it]
          new_row_val[l_it, num_varP + np.ravel_multi_index((a_it, s_it, f_it), (na, ns, nF))] = P_asS[a_it, s_it, S_it] * P_lqQ[l_it, q_it, Q_it]
        
        new_row_val[l_it, -nk:] = A_kl[:, l_it]

      A_eq = sp.bmat([[A_eq], [sp.coo_matrix(new_row_val)]])
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

  # sum over all deterministic s
  for (a_it, s_it, S_it, f_it, t_it) in product(range(na), range(ns), sDet, range(nF), range(nT)):
    q_it = F[f_it]
    Q_it = T[t_it]
    l_it = np.nonzero(conn_mat[:, S_it])[0]
    new_row[0, 1 + np.ravel_multi_index((a_it, s_it, f_it), (na, ns, nF))] -= P_asS[a_it, s_it, S_it] * P_lqQ[l_it, q_it, Q_it]


  # sum over all nondeterministic s
  for (sn_it, s_it), k_it in product(enumerate(sNDet), range(nk)):
    b_k = np.hstack([conn_mat[:, s_it], np.zeros(nl), 1, -1])
    new_row[0, num_varP + num_varX + num_varD1 + np.ravel_multi_index((sn_it, k_it), (len(sNDet), nk))] += b_k[k_it]

  A_iq = sp.bmat([[A_iq], [sp.coo_matrix(new_row)]])
  b_iq = np.hstack([b_iq, 0])

  # EQUALITY
  new_block = np.zeros((num_varD2, num_varP + num_varX + num_varD1 + num_varD2))

  A_kl = np.vstack([np.eye(nl), -np.eye(nl), np.ones([1, nl]), -np.ones([1, nl])])

  # sum over all non-determninistc s
  for (Sn_it, S_it), l_it in product(enumerate(sNDet), range(nl)):

    Sl_idx = np.ravel_multi_index((Sn_it, l_it), (len(sNDet), nl))

    # sum over saqQ
    for (a_it, s_it, f_it, t_it) in product(range(na), range(ns), range(nF), range(nT)):
      q_it = F[f_it]
      Q_it = T[t_it]
      x_idx = np.ravel_multi_index((a_it, s_it, q_it), (na, ns, nq))
      new_block[Sl_idx, num_varP + x_idx] += P_asS[a_it, s_it, S_it] * P_lqQ[l_it, q_it, Q_it]

    # sum over sk
    for ((sn_it, s_it), k_it) in product(enumerate(sNDet), range(nk)):
      sk_idx = np.ravel_multi_index((sn_it, k_it), (len(sNDet), nk))

      new_block[Sl_idx, num_varP + num_varX + num_varD1 + sk_idx] += A_kl[k_it, l_it]

  A_eq = sp.bmat([[A_eq], [sp.coo_matrix(new_block)]])
  b_eq = np.hstack([b_eq, np.zeros(num_varD2)])


  constr = Constraint(A_eq=A_eq, b_eq=b_eq, A_iq=A_iq, b_iq=b_iq)

  objective = np.zeros(num_varP + num_varX + num_varD1 + num_varD2)
  objective[0] = -1  # maximize P
  t_init = time.time()
  print(["Initiate solver: ", t_init- t] )
  sol = solve_ilp(objective, constr, J_int=[])
  t_solve = time.time()
  print(["Run solver: ", t_solve-t_init])

  if sol['status'] == 'optimal':
    return sol['x'][0], sol['x'][num_varP: num_varP+num_varX].reshape((na, ns, nF))
  else:
    print("solver returned {}".format(sol['status']))
    return -1, -1
