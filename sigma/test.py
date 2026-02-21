from sampling import *
from wavefunction import *
from observables import *
from training import *
import numpy as np
import jax.numpy as jnp
import jax

# Testing wavefunction.py
N = 5
x = np.random.normal(size=(N, 2))
model = MLP(hidden_sizes=(64, 64))
params = model.init(jax.random.PRNGKey(0), x)
y = model.apply(params, x)


def psi(params, config):
    return jnp.exp(-model.apply(params, config))


print(f"Inputs: {x} \n Outputs: {y}")
print(f"Inputs shape : {x.shape} \n Outputs shape {y.shape}")
print(f"Hidden layer widths are {model.hidden_sizes}")

cart_x = spherical_to_cartesian(x)
print(f"Cartesian x, {cart_x} with shape {cart_x.shape}")


# ===========
# testing CNN translation invariance
def test_CNN():
    N = 5
    x = np.random.normal(size=(N, 2))
    model = TI_CNN()
    params = model.init(jax.random.PRNGKey(0), x)
    y = model.apply(params, x)

    print(f"Inputs: {x} \n Outputs: {y}")
    print(f"Inputs shape : {x.shape} \n Outputs shape {y.shape}")
    test_config = x
    print(f"Original config output: {model.apply(params, test_config)}")
    for shift in range(1, N):
        shifted_config = jnp.roll(test_config, shift=shift, axis=0)
        print(
            f"Shifted config (shift={shift}) output: {model.apply(params, shifted_config)}"
        )


test_CNN()


# =================
# testing the sampler
print("Testing sampler...")
sampler = Sampler(psi, (N, 2))

nchains = 16
pos_initials = [jnp.zeros((N, 2)) for _ in range(nchains)]
seeds = jnp.arange(nchains)
var = 5

samples, acc_rate = sampler.run_many_chains(
    params, 1000, 100, 2, var, pos_initials, seeds
)

print(f"Acceptance rate: {acc_rate}")
print(samples.shape)
# =================
