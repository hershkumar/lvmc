import numpy as np
from jax import numpy as jnp
from jax import jit
from functools import partial

# File that deals with gauge configuration indexing, info retrieval, etc.

"""
Defines a gauge configuration class that stores the gauge configuration and its associated information.
"""
class GaugeConf:
    def __init__(self, L, d):
        self.L = L  # lattice size (number of vertices per dimension)
        self.d = d  # dimensionality of the lattice
        # Initialize with random SU(2) matrices (not unitary, just for placeholder)
        self.conf = jnp.array(np.random.rand(self.L**self.d * self.d, 2, 2), dtype=jnp.complex64)

    
    """
    Takes in a vertex x = (n0,n1,...,n_{d-1}) and a direction mu, and returns the gauge link at that vertex and direction.
    """
    @partial(jit, static_argnums=(0,))
    def get_link(self, x, mu):
        # Compute the site index using row-major ordering
        site_index = 0
        for j in range(self.d):
            site_index += x[j] * (self.L ** j)
        
        # Compute the link index
        link_index = self.d * site_index + mu
        
        return self.conf[link_index]
    
    
    """
    Takes in a vertex x = (n0,n1,...,n_{d-1}) and two directions, mu and nu, and returns the plaquette at that vertex and those directions.
    
    The plaquette is the product of four gauge links around a square:
    U_mu(x) * U_nu(x+mu) * U_mu^†(x+nu) * U_nu^†(x)
    
    where x+mu means incrementing the mu-th coordinate by 1 (mod L for PBC).
    """
    #TODO: make this jit compilable
    def get_plaquette(self, x, mu, nu):
        # Convert x to a list for easy modification
        x = list(x)
        
        # Helper function to get the next site in a given direction (with PBC)
        def shift(site, direction):
            shifted = site.copy()
            shifted[direction] = (shifted[direction] + 1) % self.L
            return tuple(shifted)
        
        # Get the four links of the plaquette
        # Link 1: U_mu(x)
        U1 = self.get_link(tuple(x), mu)
        
        # Link 2: U_nu(x + mu_hat)
        x_plus_mu = shift(x, mu)
        U2 = self.get_link(x_plus_mu, nu)
        
        # Link 3: U_mu^†(x + nu_hat)
        x_plus_nu = shift(x, nu)
        U3_dagger = jnp.conj(self.get_link(x_plus_nu, mu).T)
        
        # Link 4: U_nu^†(x)
        U4_dagger = jnp.conj(self.get_link(tuple(x), nu).T)
        
        # Compute the plaquette: U1 * U2 * U3^† * U4^†
        plaquette = U1 @ U2 @ U3_dagger @ U4_dagger
        
        return plaquette
    
    
    """
    Takes a configuration index, and returns the lattice information of the configuration element at that index.
    Returns (n0, n1, ..., n_{d-1}, mu).
    """
    # @partial(jit, static_argnums=(0,)) # this does jit compile, but the output is kind of annoying
    def get_info(self, index):
        # Extract direction
        mu = index % self.d
        
        # Extract site index
        site_index = index // self.d
        
        # Convert site index to coordinates
        coords = []
        for i in range(self.d):
            coords.append(site_index % self.L)
            site_index //= self.L
        
        return tuple(coords + [mu])
