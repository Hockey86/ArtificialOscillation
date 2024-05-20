import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import halfcauchy, norm
from scipy.special import logsumexp
import numpyro as npo
from jax import random
from numpyro.distributions import Normal, HalfCauchy
from numpyro.infer import MCMC, NUTS


def model1(X1, X2=None):
    # P(X2,theta|X1) = P(X2|X1,theta)P(theta)
    # P(X2|X1,theta) = N(X1*a1 + a2, s)
    # theta = {a1, a2, s}
    a1 = npo.sample('a1', Normal(0,10))
    a2 = npo.sample('a2', Normal(0,10))
    s  = npo.sample('s',  HalfCauchy(10))

    with npo.plate('N', len(X1)):
        mu = X1*a1 + a2
        npo.sample('X2', Normal(mu, s), obs=X2)


def model2(X1, X2=None):
    # let Z = 1/X1
    # P(X2,theta|X1) = P(X2|X1,theta)P(theta)
    #  = P(X2|Z,theta)P(theta)
    # P(X2|Z,theta) = N(Z*a1 + a2, s) = N(f(X1)*a1 + a2, s)
    # theta = {a1, a2, s}
    a1 = npo.sample('a1', Normal(0,10))
    a2 = npo.sample('a2', Normal(0,10))
    s  = npo.sample('s',  HalfCauchy(10))
    sz  = npo.sample('sz',  HalfCauchy(10))
    with npo.plate('N', len(X1)):
        Z = npo.sample('Z', Normal(0,sz))
        Z2 = Z+1/X1
        mu = Z2*a1 + a2
        npo.sample('X2', Normal(mu, s), obs=X2)


def run_inference(X1, X2, model, n_param):
    nuts_kernel = NUTS(model)
    mcmc = MCMC(nuts_kernel, num_warmup=500, num_samples=1000)
    rng_key = random.PRNGKey(0)
    mcmc.run(rng_key, X1, X2=X2, extra_fields=('potential_energy',))
    mcmc.print_summary()
    ll = -mcmc.get_extra_fields()['potential_energy']
    aic = 2*n_param-2*np.sum(ll)
    bic = n_param*np.log(len(X1))-2*np.sum(ll)
    print(f'average log likelihood: {np.mean(ll):.2f}')
    print(f'AIC: {aic:.2f}')
    print(f'BIC: {bic:.2f}')
"""
def run_inference_model1(x1, x2=None):
    # P(X2,theta|X1) = P(X2|X1,theta)P(theta)
    # P(X2|X1,theta) = N(X1*a1 + a2, s)
    # theta = {a1, a2, s}
    Nmcmc = 10000
    
    a1 = norm.rvs(loc=0, scale=10, size=Nmcmc)
    a2 = norm.rvs(loc=0, scale=10, size=Nmcmc)
    s  = halfcauchy.rvs(scale=10, size=Nmcmc)

    # lpr is log prior probability
    lpr = [
        norm.logpdf(a1, loc=0, scale=10),
        norm.logpdf(a2, loc=0, scale=10),
        halfcauchy.logpdf(s, scale=10), ]
    lpr = sum(lpr)

    mu = x1*a1.reshape(-1,1)+a2.reshape(-1,1)
    # ll is log likelihood
    ll = norm.logpdf(x2, loc=mu, scale=s.reshape(-1,1)).sum(axis=-1) # sum over samples

    lj = ll+lpr     # lj is log joint probability
    le = logsumexp(lj) # le is log evidence
    po = np.exp(lj-le)  # po is posterior probability
    a1_po_mean = np.sum(po*a1)
    a2_po_mean = np.sum(po*a2)
    s_po_mean = np.sum(po*s)

    df_res = pd.DataFrame(data={
        'mean':[a1_po_mean, a2_po_mean, s_po_mean], },
        index=['a1', 'a2', 's'],
    )
    print(df_res)
    return df_res


def run_inference_model2(x1, x2=None):
    # P(X2,theta|X1) = sum_Z P(X2,Z,theta|X1)
    # = sum_Z P(Z|X1,theta)P(X2|Z,theta)P(theta)
    # P(Z|X1,theta) = N(1/X1*b1, sz)
    # P(X2|Z,theta) = N(Z*a1+a2, sx)
    # theta = {a1, a2, b1, sx, sz}
    Nmcmc = 1000
    #Nmcmc_z = 500
    N = len(x1)
    
    a1 = norm.rvs(loc=0, scale=10, size=Nmcmc)
    a2 = norm.rvs(loc=0, scale=10, size=Nmcmc)
    b1 = norm.rvs(loc=0, scale=10, size=Nmcmc)
    sx  = halfcauchy.rvs(scale=10, size=Nmcmc)
    #sz  = halfcauchy.rvs(scale=10, size=Nmcmc)

    # lpr is log prior probability
    lpr = [
        norm.logpdf(a1, loc=0, scale=10),
        norm.logpdf(a2, loc=0, scale=10),
        norm.logpdf(b1, loc=0, scale=10),
        halfcauchy.logpdf(sx, scale=10),
        #halfcauchy.logpdf(sz, scale=10),
        ]
    lpr = sum(lpr)

    #mu_z = 1/x1*b1.reshape(-1,1)
    #z = norm.rvs(loc=mu_z, scale=sz.reshape(-1,1), size=(Nmcmc_z, Nmcmc, N))
    #ll_z = norm.logpdf(z, loc=mu_z, scale=sz.reshape(-1,1)).sum(axis=-1) # sum over samples
    z = 1/x1*b1.reshape(-1,1)

    mu_x = z*a1.reshape(-1,1)+a2.reshape(-1,1)
    ll_x = norm.logpdf(x2, loc=mu_x, scale=sx.reshape(-1,1)).sum(axis=-1)

    lj = lpr+ll_x#+ll_z     # lj is log joint probability
    #lj = logsumexp(lj, axis=0)  # sum over Z
    le = logsumexp(lj) # le is log evidence
    po = np.exp(lj-le)  # po is posterior probability
    a1_po_mean = np.sum(po*a1)
    a2_po_mean = np.sum(po*a2)
    b1_po_mean = np.sum(po*b1)
    sx_po_mean = np.sum(po*sx)
    #sz_po_mean = np.sum(po*sz)

    df_res = pd.DataFrame(data={
        'mean':[a1_po_mean, a2_po_mean, b1_po_mean, sx_po_mean],},#, sz_po_mean], },
        index=['a1', 'a2', 'b1', 'sx'],#, 'sz'],
    )
    print(df_res)
    return df_res
"""



def main():
    random_state = 2023
    np.random.seed(random_state)

    N = 100

    x1 = np.abs(np.random.randn(N))+1
    e1 = np.random.randn(N)/20
    e2 = np.random.randn(N)/30
    x2 = 1/(x1+e1)+2+e2

    run_inference(x1, x2, model1, 3)
    run_inference(x1, x2, model2, 3)
    #run_inference_model1(x1, x2)
    #run_inference_model2(x1, x2)

    #plt.close()
    #plt.scatter(x1, x2, c='k', s=3)
    #plt.tight_layout()
    #plt.show()


if __name__=='__main__':
    main()
