# lvmc

## Gauge Configuration Storage

Consider a lattice of $d$ dimensions (lattice dimensions), each of length $L$. Each lattice link stores a $SU(2)$ group element. Each vertex of the lattice has $d$ associated gauge links. We index a link via the index of the lattice site, and the index of the direction. We index the lattice site  $(n_0,n_1,n_2,\dots, n_{d-1})$ via:
```math
i(n_0,n_1,n_2,\dots, n_{d-1}) = n_0 + Ln_1 + \dots + L^{d-1}n_{d-1}
```
Therefore the index of the gauge link at site $(n_0,n_1,n_2,\dots, n_{d-1})$ and in the direction $\mu$, is given by:
```math
l((n_0,n_1,n_2,\dots, n_{d-1}), \mu) = d i(n_0,n_1,n_2,\dots, n_{d-1}) + \mu
```

Each element of this array is  $SU(2)$ group element, stored as a $2\times 2$ matrix of complex numbers. 





Do we want to store a second list of just plaquette traces?

