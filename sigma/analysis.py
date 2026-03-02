from plotly import graph_objects as go
from sampling import *
from wavefunction import *
from observables import *
from training import *
import numpy as np
from scipy.optimize import curve_fit
from scipy.special import kn
import csv



# # Taken from fig 5 of https://arxiv.org/pdf/2209.00098
# gsquareds = [0.5744468236973591, 0.6498215560314062,0.6698072805139187,0.6897930049964311,0.710349750178444,0.7303354746609565, 0.7497501784439686,0.7697359029264812]
# ams = [0.03983833718244804, 0.10496535796766744,0.12956120092378753,0.1558891454965358,0.18394919168591223, 0.21304849884526558,0.24353348729792149,0.2764434180138568]
# # square root of gsquareds
# gs = np.sqrt(gsquareds)
# # one over ams
# xis_expected = 1/np.array(ams)
# print(gs)
# print(xis_expected)
# # our data
# g_obs = []
# xis_observed = []




# reconstructed_fig = go.Figure()
# reconstructed_fig.add_trace(go.Scatter(x=gs, y=xis_expected, mode='markers', name='Reconstructed ξ from Literature'))
# reconstructed_fig.update_layout(title='Reconstructed Correlation Lengths ξ from Literature', xaxis_title='g', yaxis_title='ξ')
# reconstructed_fig.show()


# C, C_uncerts = np.load(f"data/L_40_g_root_.8_correlation.npy", allow_pickle=True)

# def extract_correlation_length(C):
#     L = len(C)
#     extracted = []
#     uncerts = []

#     def fit_func(n, A, xi):
#         return A* kn(0, n/xi) + A * kn(0, (L-n)/xi)
#         # return A * (np.exp(-n/xi)/np.sqrt(n) + np.exp(-(L-n)/xi)/np.sqrt(L-n))

#     for i in range(3, L):
#         data = np.arange(1, i)
#         C_data = C[1:i]

#         popt, pcov = curve_fit(fit_func, data, C_data, p0=[1, 5])

#         A_fit, xi_fit = popt
#         # get the uncertainty in the fitted parameters
#         A_uncert, xi_uncert = np.sqrt(np.diag(pcov))
#         # print(f"Fitted parameters: A = {A_fit:.4f} ± {A_uncert:.4f}, xi = {xi_fit:.4f} ± {xi_uncert:.4f}")
#         extracted.append(xi_fit)
#         uncerts.append(xi_uncert)
#     return extracted, uncerts

# xis_extracted, xis_u = extract_correlation_length(C)

# # plot this with error bars
# fig_xi = go.Figure()
# fig_xi.add_trace(go.Scatter(x=np.arange(3, len(C)), y=xis_extracted, error_y=dict(type='data', array=xis_u), mode='markers', name='Extracted ξ'))
# fig_xi.update_layout(title='Extracted Correlation Lengths ξ', xaxis_title='n', yaxis_title='ξ')
# fig_xi.show()  


def perturbative(g_squared):
    beta = 1/g_squared
    return 1/(128*np.pi*beta * np.exp(-2 * np.pi * beta))


g_squared = [1, .95, .9, .85, .8, .77, .75,.87]
xi_obs = [2.0085730585630297, 2.3696835989225935,2.814879144995261, 3.5144897593139173,4.756214785571526,5.600071408138509,6.912242240903248, 3.2531248756596773]
xi_uncert = [0.0054205592782067364, 0.0037734950620795646 ,0.007525188902512143,0.013275788290214481,0.012411322463477679,0.024057648847475164,0.019325924397643798,0.009299356468873555]

fig = go.Figure()
fig.add_trace(go.Scatter(x=g_squared, y=xi_obs, error_y=dict(type='data', array=xi_uncert), mode='markers', name='Observed ξ'))
# add the perturbative curve
g_squared_continuous = np.linspace(0.75, 1, 100)
xi_perturbative = perturbative(g_squared_continuous)
fig.add_trace(go.Scatter(x=g_squared_continuous, y=xi_perturbative, mode='lines', name='Perturbative ξ'))
fig.update_layout(title='Observed Correlation Lengths ξ vs g^2', xaxis_title='g^2', yaxis_title='ξ')
fig.show()


# save the observed correlation lengths to a csv
with open('data/observed_correlation_lengths.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['g_squared', 'xi_observed', 'xi_uncertainty'])
    for g2, xi, xi_u in zip(g_squared, xi_obs, xi_uncert):
        writer.writerow([g2, xi, xi_u])