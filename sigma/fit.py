import numpy as np 
from matplotlib import pyplot as plt
from scipy.special import kn
from scipy.optimize import curve_fit

def extract_correlation_length_old(C, C_uncert, fit_type="exponential"):
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
        A_fit, xi_fit = popt # get the uncertainty in the fitted parameters 
        A_uncert, xi_uncert = np.sqrt(np.diag(pcov)) # 
        print(f"Fitted parameters: A = {A_fit:.4f} ± {A_uncert:.4f}, xi = {xi_fit:.4f} ± {xi_uncert:.4f}")
        extracted.append(xi_fit)
        uncerts.append(xi_uncert)
    return extracted, uncerts

def extract_correlation_length(C, C_uncert, fit_type="exponential", alpha=0.05):
    C = np.asarray(C, dtype=float)
    C_uncert = np.asarray(C_uncert, dtype=float)

    L = len(C)
    extracted = []
    uncerts = []

    def fit_func(n, A, xi):
        n = np.asarray(n, dtype=float)
        if fit_type == "exponential":
            return A * (np.exp(-n/xi)/np.sqrt(n) + np.exp(-(L-n)/xi)/np.sqrt(L-n))
        elif fit_type == "bessel":
            return A * (kn(0, n/xi) + kn(0, (L-n)/xi))
        else:
            raise ValueError("Invalid fit type. Must be 'exponential' or 'bessel'.")

    for i in range(3, L):
        sl = slice(1, i)
        data = np.arange(1, i, dtype=float)
        C_data = C[sl]

        if C_uncert.ndim == 1:
            sigma_data = C_uncert[sl]
        elif C_uncert.ndim == 2:
            sigma_data = C_uncert[sl, sl]
            sigma_data = 0.5 * (sigma_data + sigma_data.T)

            # shrinkage regularization
            D = np.diag(np.diag(sigma_data))
            sigma_data = (1 - alpha) * sigma_data + alpha * D

            # small ridge
            eps = 1e-12 * np.trace(sigma_data) / sigma_data.shape[0]
            sigma_data = sigma_data + eps * np.eye(sigma_data.shape[0])
        else:
            raise ValueError("C_uncert must be 1D or 2D.")

        A0 = C_data[0] if np.isfinite(C_data[0]) and C_data[0] != 0 else 1.0
        xi0 = 5.0

        try:
            popt, pcov = curve_fit(
                fit_func,
                data,
                C_data,
                sigma=sigma_data,
                p0=[A0, xi0],
                absolute_sigma=True,
                maxfev=10000,
            )
            xi_fit = popt[1]
            xi_uncert = np.sqrt(np.diag(pcov))[1]
        except Exception:
            xi_fit = np.nan
            xi_uncert = np.nan

        extracted.append(xi_fit)
        uncerts.append(xi_uncert)

    return np.asarray(extracted), np.asarray(uncerts)

def extract_correlation_length_from_file(filename, fit_type="exponential"):
    C, C_uncert = np.load(filename, allow_pickle=True)
    return extract_correlation_length(C, C_uncert, fit_type)