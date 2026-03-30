import numpy as np 
from matplotlib import pyplot as plt
from scipy.special import kn
from scipy.optimize import curve_fit


def extract_correlation_length(C, C_uncert, fit_type="exponential"):
    L = len(C)
    extracted = []
    uncerts = []
    def fit_func(n, A, xi):
        if fit_type == "exponential":
            return A * (np.exp(-n/xi)/np.sqrt(n) + np.exp(-(L-n)/xi)/np.sqrt(L-n))
        elif fit_type == "bessel":
            return A * (kn(0, n/xi) + kn(0, (L-n)/xi))
        else:
            raise ValueError("Invalid fit type. Must be 'exponential' or 'bessel'.")
    for i in range(3, L):
        data = np.arange(1, i)
        C_data = C[1:i]
        C_uncert_data = C_uncert[1:i]

        popt, pcov = curve_fit(fit_func, data, C_data, sigma=C_uncert_data, p0=[1, 5])

        A_fit, xi_fit = popt
        # get the uncertainty in the fitted parameters
        A_uncert, xi_uncert = np.sqrt(np.diag(pcov))
        # print(f"Fitted parameters: A = {A_fit:.4f} ± {A_uncert:.4f}, xi = {xi_fit:.4f} ± {xi_uncert:.4f}")
        extracted.append(xi_fit)
        uncerts.append(xi_uncert)
    return extracted, uncerts


def extract_correlation_length_from_file(filename, fit_type="exponential"):
    C, C_uncert = np.load(filename, allow_pickle=True)
    return extract_correlation_length(C, C_uncert, fit_type)