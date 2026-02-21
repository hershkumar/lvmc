from jax.tree_util import tree_map
from jax.numpy import zeros_like, mean


def pytree_mult(scalar, pytree):
    """
    Multiplies a pytree by a scalar, elementwise.
    """
    return tree_map(lambda x: scalar * x, pytree)


def pytree_add(a, b):
    """
    Adds two pytrees elementwise. The two pytrees must have the same structure.
    """
    return tree_map(lambda x, y: x + y, a, b)


def pytree_mean(pytree_batched):
    """
    Returns a pytree whose elements are the mean over the first axis of the input pytree elements.
    """
    return tree_map(lambda x: mean(x, axis=0), pytree_batched)


def pytree_zeros_like(pytree):
    """
    Returns a pytree with the same structure as the input pytree, but with all elements replaced by zeros.
    """
    return tree_map(zeros_like, pytree)
