from sampling import *
from wavefunction import *
from observables import *
from training import *
import numpy as np
import jax.numpy as jnp
import jax

# Testing wavefunction.py
N = 5
x = np.random.normal(size=(N,2))
model = psi(hidden_sizes=(64,64))
params = model.init(jax.random.PRNGKey(0), x)
y = model.apply(params,x)

print(f'Inputs: {x} \n Outputs: {y}')
print(f'Inputs shape : {x.shape} \n Outputs shape {y.shape}')

cart_x = spherical_to_cartesian(x)
print(f'Cartesian x, {cart_x} with shape {cart_x.shape}')
