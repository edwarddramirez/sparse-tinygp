import numpy as np # for numerical operations
import matplotlib.pyplot as plt # for plotting

import jax # for faster numerical operations
import jax.numpy as jnp # for numpy-like syntax
from jax.scipy import stats
from jax import jit

from tinygp import kernels, GaussianProcess # for kernels

import numpyro # for inference and probabilistic programming
import numpyro.distributions as dist
from numpyro.infer.autoguide import AutoMultivariateNormal
from numpyro.infer import Trace_ELBO

from tqdm import tqdm # for progress bar
from collections import namedtuple # for named tuples

from scipy.stats import chi2

def poisson_interval(k, alpha=0.32): 
    """ 
    Uses chi2 to get the poisson interval.
    """
    a = alpha
    low, high = (chi2.ppf(a/2, 2*k) / 2, chi2.ppf(1-a/2, 2*k + 2) / 2)
    if k == 0: 
        low = 0.0
    return k - low, high - k

# poisson likelihood function
def log_like_poisson(mu, data):
    return dist.discrete.Poisson(mu).log_prob(data)

def load_svi(model, lr=0.01, num_particles = 16):
    guide = AutoMultivariateNormal(model)
    optim = numpyro.optim.Adam(lr)
    svi = numpyro.infer.SVI(model, guide, optim, Trace_ELBO(num_particles))
    return svi, guide

def load_kernel(before_fit = True, params = None):
    if before_fit:
        # define Gaussian Process prior on log_rate (zero mean and ExpSquared kernel)
        amp = numpyro.param(
            "amp", jnp.ones(()), constraint=dist.constraints.positive
        )
        scale = numpyro.param(
            "scale", 5. * jnp.ones(()), constraint=dist.constraints.positive
        )
    else:
        amp = params['amp']
        scale = params['scale']
    return amp**2. * kernels.ExpSquared(scale)

def get_gp_samples_at_x_from_u(rng_key, num_samples, pred, params, x, xu):
    '''
    Vectorized GP sampling at x from u
    '''
    # generate keys for gp samples
    rng_key_vec = jax.random.split(rng_key, num_samples)

    # generate data in a vectorizable way
    # key is to vectorize the operation with respect to a vector of rng_keys
    def generate_data(rng_key):
        # generate kernel parameters with best-fit GP parameters
        base_kernel = load_kernel(before_fit = False, params = params)
        gp_u = GaussianProcess(base_kernel, xu, diag=1e-3)

        gp_u_key, gp_key = jax.random.split(rng_key, 2)
        log_rate_u = pred(gp_u_key)['log_rate_u'].T
        log_rate_u = jnp.squeeze(log_rate_u, axis = -1) # shape mismatch otherwise (shape_ll ~ (shape_x, shape_x))
        _, gp_x = gp_u.condition(log_rate_u, x, diag = 1e-3) # p(x|u)
        log_rate = gp_x.sample(gp_key)
        return log_rate

    generate_data_vec = jax.vmap(generate_data)

    def body_fn():
        rng_key_vec = jax.random.split(rng_key, num_samples)
        log_rate = generate_data_vec(rng_key_vec)
        return log_rate
    
    return jit(body_fn)()

def svi_loop(rng_key, num_steps, svi, x, xu, y, gp_rng_key = jax.random.PRNGKey(0)):
    '''
    Simple Loop for SVI optimization
    '''
    # svi update function
    def body_fn(i, svi_state):
        gp_rng_key = jax.random.split(svi_state.rng_key)[-1]
        svi_state, train_loss = svi.update(svi_state, x, xu, y, gp_rng_key)
        return svi_state, train_loss
    
    # initial svi state
    svi_state = svi.init(rng_key, x, xu, y, gp_rng_key)
    
    # loop over num_steps
    losses = [] 
    for n in tqdm(range(num_steps)):
        svi_state, loss = jit(body_fn)(n, svi_state)
        losses.append(loss) 
        
    params = svi.get_params(svi_state)
    
    SVIRunResult = namedtuple("SVIRunResult", ["params", "state", "losses"])
    return SVIRunResult(params, svi_state, losses)