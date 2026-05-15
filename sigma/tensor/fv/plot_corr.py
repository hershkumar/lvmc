import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# User settings
# ============================================================

csv_specs = [
    {
        "path": "100_100_wavefunction_corr (2).csv",
        "r_offset": 0,
        "label": "wavefunction corr",
    },
    # Add more CSVs like this:
    # {
    #     "path": "another_correlator.csv",
    #     "r_offset": 10,
    #     "label": "another corr",
    # },
]

plot_abs = True          # plot log|C(r)| instead of log C(r)
use_error_bars = True    # requires an "err" column
save_fig = False
out_path = "log_correlator.png"

# ============================================================
# Loader
# ============================================================

def load_correlator_csv(path):
    path = Path(path)
    df = pd.read_csv(path)

    required = {"r", "C"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    df = df.copy()
    df["r"] = pd.to_numeric(df["r"], errors="coerce")
    df["C"] = pd.to_numeric(df["C"], errors="coerce")

    if "err" in df.columns:
        df["err"] = pd.to_numeric(df["err"], errors="coerce")
    else:
        df["err"] = np.nan

    df = df.dropna(subset=["r", "C"])
    df = df.sort_values("r")

    return df

# ============================================================
# Plot
# ============================================================

plt.figure(figsize=(8, 5))

for spec in csv_specs:
    csv_path = Path(spec["path"])
    r_offset = spec.get("r_offset", 0)
    label = spec.get("label", csv_path.stem)

    df = load_correlator_csv(csv_path)

    r = df["r"].to_numpy(dtype=float) + r_offset
    C = df["C"].to_numpy(dtype=float)
    err = df["err"].to_numpy(dtype=float)

    if plot_abs:
        y_raw = np.abs(C)
        ylabel = r"$\log |C(r)|$"
    else:
        y_raw = C
        ylabel = r"$\log C(r)$"

    valid = np.isfinite(r) & np.isfinite(y_raw) & (y_raw > 0)

    r_plot = r[valid]
    y_plot = np.log(y_raw[valid])

    label = f"{label}, offset={r_offset}"

    if use_error_bars and np.any(np.isfinite(err[valid])):
        # Error propagation:
        # y = log(C), so dy ≈ err / |C|
        yerr = err[valid] / y_raw[valid]

        plt.errorbar(
            r_plot,
            y_plot,
            yerr=yerr,
            marker="o",
            linestyle="-",
            capsize=3,
            label=label,
        )
    else:
        plt.plot(
            r_plot,
            y_plot,
            marker="o",
            linestyle="-",
            label=label,
        )

plt.xlabel(r"offset separation $r + r_0$")
plt.ylabel(ylabel)
plt.title("Log correlator vs offset separation")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

if save_fig:
    plt.savefig(out_path, dpi=200)

plt.show()
