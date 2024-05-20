"""
MCMC limitation: the functions cannot be too complex which then affects sampling
P(Zt | X1_{t-H:t}, X2_{t-H-1:t-1}, Z_{t-H-1:t-1})
P(X2t | X1_{t-H-1:t-1}, X2_{t-H-1:t-1}, Z_{t-H:t})
"""
from functools import partial
import numpy as np
import pandas as pd
import arviz as az
import jax
import jax.numpy as jnp
from jax.scipy.special import expit as sigmoid
import numpyro as npo
from numpyro.contrib.control_flow import scan
from numpyro.infer import MCMC, NUTS, Predictive, SVI, Trace_ELBO
from numpyro.optim import Adam
import numpyro.distributions as dist
import matplotlib.pyplot as plt


def generate_data(T, random_state=None):
    """
    """
    np.random.seed(random_state)
    X1 = np.random.choice([-1,0,1], T).astype(float)
    X1n = X1+np.random.randn(T)*0.05
    Z = np.cumsum(X1n)
    X2 = 100-Z+np.random.randn(T)*0.1
    df = pd.DataFrame(data={'X1':X1, 'Z':Z, 'X2':X2}) #TODO multidimensional X1 and Z and X2
    return df


class Model:
    def __init__(self, model_func, inference_method='mcmc', num_samples=1000, num_burnin=500, verbose=True, random_state=None):
        self.model_func = model_func
        self.inference_method = inference_method
        self.num_samples = num_samples
        self.num_burnin = num_burnin
        self.verbose = verbose
        self.random_state = random_state

    def fit(self, X, y):
        """
        """
        if self.inference_method=='mcmc':
            nuts_kernel = NUTS(self.model_func)
            self.mcmc = MCMC(nuts_kernel, num_samples=self.num_samples, num_warmup=self.num_burnin, progress_bar=self.verbose)
            rng_key = jax.random.PRNGKey(self.random_state)
            self.mcmc.run(rng_key, jnp.array(X), jnp.array(y))
            if self.verbose:
                self.mcmc.print_summary()
            self.posterior = self.mcmc.get_samples()

            # convert to arviz InferenceData
            _, rng_key1 = jax.random.split(rng_key)
            _, rng_key2 = jax.random.split(rng_key1)
            self.az_inferencedata = az.from_numpyro(
                self.mcmc,
                prior=Predictive(self.model_func, num_samples=self.num_samples)(
                    rng_key1, jnp.array(X)),
                posterior_predictive=Predictive(self.model_func, self.posterior)(
                    rng_key2, jnp.array(X))
            )
        else:
            optimizer = Adam(step_size=0.0001)
            svi = SVI(self.model_func, self.guide_func, optimizer, loss=Trace_ELBO())
            rng_key = jax.random.PRNGKey(self.random_state)
            svi_result = svi.run(rng_key, self.num_samples, jnp.array(X), jnp.array(y), progress_bar=self.verbose)
            self.posterior = svi_result.params

            #TODO convert to arviz InferenceData

        return self

    def predict(self, X, return_az=False):
        """
        """
        rng_key = jax.random.PRNGKey(self.random_state)
        _, rng_key = jax.random.split(rng_key)
        _, rng_key = jax.random.split(rng_key)
        pred = Predictive(
                self.model_func if self.inference_method=='mcmc' else self.guide_func,
                self.posterior)(
                rng_key, jnp.array(X))
        #TODO if return_az:
        return pred

    def score(self, score_type='loo'):
        if score_type=='waic':
            score_ = az.waic(self.az_inferencedata).waic
        else:
            score_ = az.loo(self.az_inferencedata).loo
        return score_


def f(X, params):
    res = jnp.dot(X,params['beta'])+params['beta0']

    #z1 = jax.nn.relu(jnp.matmul(X, params['w1']))
    #z2 = jax.nn.relu(jnp.matmul(z1, params['w2']))
    #res = jnp.matmul(z2, params['w3'])[:,0]
    return res


def model1_func(span, X1, X2=None):
    """
    """
    l1_scale = npo.sample('l1_scale', dist.HalfCauchy(scale=100))
    sigma2 = npo.sample('sigma2', dist.HalfCauchy(scale=100))
    beta_V0 = npo.sample('beta_V0', dist.Normal(loc=0,scale=100))
    beta_V = npo.sample('beta_V', dist.Laplace(loc=0,scale=l1_scale), sample_shape=(span*(1+1),))

    def model_onestep(history, t):
        X1_history = history[0]
        X2_history = history[1]

        # P(X2t | X1_{t-H-1:t-1}, X2_{t-H-1:t-1})
        xx = jnp.r_[X1_history, X2_history].reshape(1,-1)
        X2t = f(xx, {'beta':beta_V, 'beta0':beta_V0})[0]

        new_X1_history = jnp.r_[X1_history[1:], X1[t]]
        new_X2_history = jnp.r_[X2_history[1:], X2t]

        return jnp.array([new_X1_history, new_X2_history]), X2t

    history0 = jnp.zeros((1+1, span))
    _, pred = scan(model_onestep, history0, np.arange(len(X1)))
    npo.sample('X2', dist.Normal(pred, sigma2), obs=X2)


def model2_func(span, X1, X2=None):
    """
    """

    D = span*(1+1+1)
    l1_scale = npo.sample('l1_scale', dist.HalfCauchy(scale=100))
    sigma2 = npo.sample('sigma2', dist.HalfCauchy(scale=100))
    beta_H0 = npo.sample('beta_H0', dist.Normal(loc=0,scale=100))
    beta_V0 = npo.sample('beta_V0', dist.Normal(loc=0,scale=100))
    beta_H = npo.sample('beta_H', dist.Laplace(loc=0,scale=l1_scale), sample_shape=(D,))
    beta_V = npo.sample('beta_V', dist.Laplace(loc=0,scale=l1_scale), sample_shape=(D,))
    """

    DH = D*2
    DY = 1
    l1_scale = npo.sample('l1_scale', dist.HalfCauchy(scale=100))
    w1_z = npo.sample("w1_z", dist.Laplace(loc=0, scale=l1_scale), sample_shape=(D,DH))
    w2_z = npo.sample("w2_z", dist.Laplace(loc=0, scale=l1_scale), sample_shape=(DH,DH))
    w3_z = npo.sample("w3_z", dist.Laplace(loc=0, scale=l1_scale), sample_shape=(DH,DY))
    #sigma2_z = npo.sample('sigma2_z', dist.HalfCauchy(scale=100))
    w1 = npo.sample("w1", dist.Laplace(loc=0, scale=l1_scale), sample_shape=(D,DH))
    w2 = npo.sample("w2", dist.Laplace(loc=0, scale=l1_scale), sample_shape=(DH,DH))
    w3 = npo.sample("w3", dist.Laplace(loc=0, scale=l1_scale), sample_shape=(DH,DY))
    sigma2 = npo.sample('sigma2', dist.HalfCauchy(scale=100))
    """

    def model_onestep(history, t):
        X1_history = history[0]
        Z_history = history[1]
        X2_history = history[2]

        new_X1_history = jnp.r_[X1_history[1:], X1[t]]

        # P(Zt | X1_{t-H:t}, X2_{t-H-1:t-1}, Z_{t-H-1:t-1})
        xx = jnp.r_[new_X1_history, X2_history, Z_history].reshape(1,-1)
        Zt = f(xx, {'beta':beta_H, 'beta0':beta_H0})[0]
        #Zt = f(xx, {'w1':w1_z, 'w2':w2_z, 'w3':w3_z})[0]

        new_Z_history = jnp.r_[Z_history[1:], Zt]

        # P(X2t | X1_{t-H-1:t-1}, X2_{t-H-1:t-1}, Z_{t-H:t})
        xx = jnp.r_[X1_history, X2_history, new_Z_history].reshape(1,-1)
        X2t = f(xx, {'beta':beta_V, 'beta0':beta_V0})[0]
        #X2t = f(xx, {'w1':w1, 'w2':w2, 'w3':w3})[0]

        new_X2_history = jnp.r_[X2_history[1:], X2t]

        return (new_X1_history, new_Z_history, new_X2_history), (Zt, X2t)

    history0 = (jnp.zeros(span),)*3
    _, pred = scan(model_onestep, history0, np.arange(len(X1)))
    npo.deterministic('Z', pred[0])
    npo.sample('X2', dist.Normal(pred[1], sigma2), obs=X2)

    
def model2_guide_func(data):
    alpha_q = npo.param("alpha_q", 15., constraint=constraints.positive)

    l1_scale = npo.sample('l1_scale', dist.Normal(l1_scale_mean, l1_scale_std), constraint=constraints.positive)
    sigma2 = npo.sample('sigma2', dist.HalfCauchy(scale=100))
    beta_H0 = npo.sample('beta_H0', dist.Normal(loc=0,scale=100))
    beta_V0 = npo.sample('beta_V0', dist.Normal(loc=0,scale=100))
    beta_H = npo.sample('beta_H', dist.Laplace(loc=0,scale=l1_scale), sample_shape=(D,))
    beta_V = npo.sample('beta_V', dist.Laplace(loc=0,scale=l1_scale), sample_shape=(D,))



if __name__=='__main__':
    ## prepare data

    T = 100  # total time steps
    random_state = 2022
    df = generate_data(T, random_state=random_state)

    ## fit model
    span = 3

    model = Model( partial(model2_func, span),
        num_samples=2000, num_burnin=1000,
        random_state=random_state)
    model.fit(df.X1, df.X2)
    print(model.score())

    ## predict

    T = T*2
    df = generate_data(T, random_state=random_state+1)
    pred = model.predict(df.X1)
    X2_pred = pred['X2']
    Z_pred = pred['Z']

    ## plot

    plt.close()
    fig = plt.figure()
    ax = fig.add_subplot(211)
    ax.fill_between(np.arange(T),np.percentile(Z_pred,2.5,axis=0),np.percentile(Z_pred,97.5,axis=0),color='r',alpha=0.3)
    ax.plot(np.mean(Z_pred, axis=0),c='r')
    ax.plot(df.Z,c='k')
    ax.set_ylabel('Z')
    ax = fig.add_subplot(212)
    ax.fill_between(np.arange(T),np.percentile(X2_pred,2.5,axis=0),np.percentile(X2_pred,97.5,axis=0),color='r',alpha=0.3)
    ax.plot(np.mean(X2_pred, axis=0),c='r')
    ax.plot(df.X2,c='k')
    ax.set_ylabel('X2')
    plt.tight_layout()
    plt.show()
    import pdb;pdb.set_trace()
