from plotly import graph_objects as go
from sampling import *
from wavefunction import *
from observables import *
from training import *
import numpy as np




# Taken from fig 5 of https://arxiv.org/pdf/2209.00098
gsquareds = [0.5744468236973591, 0.6498215560314062,0.6698072805139187,0.6897930049964311,0.710349750178444,0.7303354746609565, 0.7497501784439686,0.7697359029264812]
ams = [0.03983833718244804, 0.10496535796766744,0.12956120092378753,0.1558891454965358,0.18394919168591223, 0.21304849884526558,0.24353348729792149,0.2764434180138568]
# square root of gsquareds
gs = np.sqrt(gsquareds)
# one over ams
xis_expected = 1/np.array(ams)

# our data





reconstructed_fig = go.Figure()
reconstructed_fig.add_trace(go.Scatter(x=gs, y=xis_expected, mode='markers', name='Reconstructed ξ from Literature'))
reconstructed_fig.update_layout(title='Reconstructed Correlation Lengths ξ from Literature', xaxis_title='g', yaxis_title='ξ')
reconstructed_fig.show()

