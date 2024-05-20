from itertools import combinations
import os
import subprocess
import numpy as np
from scipy.stats import pearsonr
import scipy.io as sio
r_bin_path = r'D:\software\R-4.2.0\bin'


def fci(data, n_jobs=1, verbose=False):
    """
    """
    nodes = sorted(data.keys())
    D = len(nodes)
    N = len(data[nodes[0]])
    corr_mat = np.ones((D,D))
    for i,j in combinations(range(D),2):
        corr_mat[i,j] = corr_mat[j,i] = pearsonr(data[nodes[i]], data[nodes[j]])[0]

    folder = os.getcwd()
    mat_fname = 'tmp.mat'
    sio.savemat(mat_fname, {'corr.mat':corr_mat, 'n':N, 'nodes':nodes})
    
    res_fname = 'tmp_res.mat'
    rcode = f"""suppressMessages(library(pcalg))
suppressMessages(library(R.matlab))

mat <- readMat(file.path(r'({folder})', '{mat_fname}'))
suffStat <- list(C = mat$corr.mat, n = mat$n)
res <- fci(suffStat, gaussCItest, 0.1, mat$nodes,    #TODO CI test for generic input?
           fixedGaps = NULL, fixedEdges = NULL,      #TODO
           numCores = {n_jobs}, selectionBias = TRUE,
           verbose = {str(verbose).upper()})
summary(res)
#jci = c("0","1","12","123"), contextVars = NULL, 

# skel.method = c("stable", "original", "stable.fast"),
# type = c("normal", "anytime", "adaptive"),
# NAdelete = TRUE, m.max = Inf, pdsep.max = Inf,
# rules = rep(TRUE, 10), doPdsep = TRUE, biCC = FALSE,
# conservative = FALSE, maj.rule = FALSE,

writeMat(file.path(r'({folder})', '{res_fname}'), amat=res@amat)
"""
    rcode_path = 'tmp.R'
    with open(rcode_path, 'w') as f:
        f.write(rcode)

    subprocess.run(
            [os.path.join(r_bin_path, 'Rscript'), rcode_path],
            stdout=None if verbose else subprocess.DEVNULL, check=True)

    #read
    amat = sio.loadmat(res_fname)['amat'].astype(int)
    
    # delete tmp files
    os.remove(rcode_path)
    os.remove(mat_fname)
    os.remove(res_fname)

    return amat

