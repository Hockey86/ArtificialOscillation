"""
defines a PAG class
generate data from true DAG* with hidden variables
    only observational
*true DAG can be A->B C->D
                    ^    ^
                    U----+
run R::pcalg::FCI --> a PAG
run my algorithm (PAG+node-wise intervention) to find the optimal actions
apply the optimal actions
run R::pcalg::FCI-JCI --> a PAG
compare derived graph with the true DAG
"""
import numpy as np
from causal_diagram import DAG#, PAG
from causal_inference import fci


def funcB(A=0, U=0):
    return A+U+np.random.randn(*A.shape)*0.1


def funcD(C=0, U=0):
    return C+U+np.random.randn(*C.shape)*0.1


def find_optimal_actions(pag):
    pass


if __name__=='__main__':
    ## generate data from true DAG with hidden variables

    V = { 'A':np.random.randn(1),
          'B':np.random.randn(1),
          'C':np.random.randn(1),
          'D':np.random.randn(1),
          'U':np.random.randn(1), }
    F = { ('B','A','U'):funcB,
          ('D','C','U'):funcD }
    true_dag = DAG(V, F)

    np.random.seed(2022)
    N = 100
    data_true_full = true_dag.run(N=N, reset_root=True)
    # hide U
    data_true = {k:v for k,v in data_true_full.items() if k!='U'}

    ## run R::pcalg::FCI --> a PAG
    infered_pag_with_full = fci(data_true_full, verbose=True)#TODO convert the returned adj mat into PAG!
    infered_pag = fci(data_true, verbose=True)

    ## run my algorithm (PAG+node-wise intervention) to find the optimal actions
    actions = find_optimal_actions(infered_pag)

    ## apply the optimal actions
