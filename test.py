from configurations import *
from equivariant import *


# testbed for gauge configurations
L = 5
d = 2

g = GaugeConf(L=L, d=d)

print(g.conf.shape)

for a in range(L):
    for b in range(L):
        for mu in range(d):
            print(f"Link at vertex ({a},{b}) and direction {mu}:")
            print(g.get_link((a,b),mu))


# also check the inverse direction
for i in range(g.conf.shape[0]):
    print(g.get_info(i))


print("Testing plaquette computation:")
print(g.get_plaquette((2,3), 0, 1))