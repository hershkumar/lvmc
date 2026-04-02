import traceback
import numpy as np
import gvar as gv
import ipywidgets as widgets
from IPython.display import display
import plotly.graph_objects as go
from fit import *

# ============================================================
# Utilities
# ============================================================

def regularize_cov(cov, rel_cut=1e-5):
    cov = np.asarray(cov, dtype=np.float64)
    cov = 0.5 * (cov + cov.T)

    evals, evecs = np.linalg.eigh(cov)
    lam_max = np.max(evals)
    cut = rel_cut * lam_max
    evals_reg = np.clip(evals, cut, None)

    cov_reg = (evecs * evals_reg) @ evecs.T
    cond = np.linalg.cond(cov_reg)
    return cov_reg, cond


def compute_xi_scan(C, cov, fit_type="exponential", rel_cut=1e-5):
    """
    Recompute the full xi scan using the existing pipeline:
        xis, xis_u = extract_correlation_length(C, cov_reg, fit_type)
    """
    cov_reg, cond = regularize_cov(cov, rel_cut=rel_cut)

    try:
        xis, xis_u = extract_correlation_length(C, cov_reg, fit_type)
    except Exception:
        print("Failure inside extract_correlation_length(...)")
        traceback.print_exc()
        raise

    xis = np.asarray(xis, dtype=np.float64)
    xis_u = np.asarray(xis_u, dtype=np.float64)

    L = len(C)
    xs = np.arange(3, L)

    n = min(len(xs), len(xis), len(xis_u))
    xs = xs[:n]
    xis = xis[:n]
    xis_u = xis_u[:n]

    return xs, xis, xis_u, cond


# ============================================================
# Interactive FigureWidget UI
# ============================================================

def fitting_widget(
    C,
    cov,
    initial_fit_type="exponential",
    initial_rel_cut=1e-5,
    initial_plateau=(22, 24),
):
    C = np.asarray(C, dtype=np.float64)
    L = len(C)

    x_min = 3
    x_max = max(3, L - 1)

    p0_default = min(max(initial_plateau[0], x_min), max(x_min, L - 2))
    p1_default = min(max(initial_plateau[1], x_min), x_max)
    if p1_default < p0_default:
        p1_default = p0_default

    # ---------- controls ----------
    fit_dropdown = widgets.Dropdown(
        options=["exponential", "bessel"],
        value=initial_fit_type,
        description="fit",
        layout=widgets.Layout(width="220px"),
    )

    logcut_slider = widgets.FloatSlider(
        value=np.log10(initial_rel_cut),
        min=-12,
        max=-2,
        step=0.5,
        description="log10 cut",
        continuous_update=False,
        layout=widgets.Layout(width="420px"),
    )

    plateau_start = widgets.IntSlider(
        value=p0_default,
        min=x_min,
        max=max(x_min, L - 2),
        step=1,
        description="plateau i",
        continuous_update=False,
        layout=widgets.Layout(width="420px"),
    )

    plateau_end = widgets.IntSlider(
        value=p1_default,
        min=x_min,
        max=x_max,
        step=1,
        description="plateau j",
        continuous_update=False,
        layout=widgets.Layout(width="420px"),
    )

    extract_button = widgets.Button(
        description="Extract plateau ξ",
        button_style="",
    )

    status_text = widgets.HTML(value="")
    result_text = widgets.HTML(value="")

    # ---------- figure ----------
    fig_xi = go.FigureWidget()

    fig_xi.add_scatter(
        x=[],
        y=[],
        error_y=dict(type="data", array=[], visible=True),
        mode="markers+lines",
        name="Extracted ξ",
    )

    fig_xi.add_scatter(
        x=[],
        y=[],
        mode="lines",
        name="Plateau mean",
    )

    fig_xi.update_layout(
        title="Extracted Correlation Lengths ξ",
        xaxis_title="Fitting Range End n",
        yaxis_title="Extracted ξ",
        template="plotly_white",
        height=500,
        shapes=[],
    )

    state = {
        "xs": None,
        "xis": None,
        "xis_u": None,
        "cond": None,
        "fit_type": None,
    }


    result_state = {"xi": None}


    def _clear_plot():
        with fig_xi.batch_update():
            fig_xi.data[0].x = []
            fig_xi.data[0].y = []
            fig_xi.data[0].error_y.array = []
            fig_xi.data[1].x = []
            fig_xi.data[1].y = []
            fig_xi.update_layout(shapes=[])

    def _set_plateau_rectangle(i0, i1):
        # Explicit shape avoids the FigureWidget/add_vrect bug
        fig_xi.update_layout(
            shapes=[
                dict(
                    type="rect",
                    xref="x",
                    yref="paper",
                    x0=i0 - 0.5,
                    x1=i1 + 0.5,
                    y0=0.0,
                    y1=1.0,
                    fillcolor="LightSkyBlue",
                    opacity=0.12,
                    line=dict(width=0),
                    layer="below",
                )
            ]
        )

    def _clear_plateau_rectangle():
        fig_xi.update_layout(shapes=[])

    def _set_plateau_visuals(show_mean_line=True):
        xs = state["xs"]
        xis = state["xis"]
        xis_u = state["xis_u"]

        i0 = int(plateau_start.value)
        i1 = int(plateau_end.value)

        with fig_xi.batch_update():
            if i1 >= i0:
                _set_plateau_rectangle(i0, i1)
            else:
                _clear_plateau_rectangle()

            if not show_mean_line or xs is None or xis is None or xis_u is None or i1 < i0:
                fig_xi.data[1].x = []
                fig_xi.data[1].y = []
                return

            mask = (xs >= i0) & (xs <= i1)
            if np.any(mask):
                gv_xis = gv.gvar(xis, xis_u)
                plateau = gv_xis[mask]
                extracted_xi = np.mean(plateau)

                fig_xi.data[1].x = [xs[0], xs[-1]]
                fig_xi.data[1].y = [extracted_xi.mean, extracted_xi.mean]
            else:
                fig_xi.data[1].x = []
                fig_xi.data[1].y = []

    def update_plot(*_):
        fit_type = fit_dropdown.value
        rel_cut = 10.0 ** float(logcut_slider.value)

        result_text.value = ""

        try:
            xs, xis, xis_u, cond = compute_xi_scan(
                C,
                cov,
                fit_type=fit_type,
                rel_cut=rel_cut,
            )

            state["xs"] = xs
            state["xis"] = xis
            state["xis_u"] = xis_u
            state["cond"] = cond
            state["fit_type"] = fit_type

            with fig_xi.batch_update():
                fig_xi.data[0].x = xs
                fig_xi.data[0].y = xis
                fig_xi.data[0].error_y.array = xis_u

            _set_plateau_visuals(show_mean_line=False)

            status_text.value = (
                f"<b>fit type</b>: {fit_type} &nbsp;&nbsp; "
                f"<b>cond(cov)</b>: {cond:.3e} &nbsp;&nbsp; "
                f"<b>points</b>: {len(xs)}"
            )

        except Exception as e:
            state["xs"] = None
            state["xis"] = None
            state["xis_u"] = None
            state["cond"] = None
            state["fit_type"] = fit_type

            _clear_plot()
            traceback.print_exc()

            status_text.value = (
                f"<span style='color:red'><b>Update failed</b>: "
                f"{type(e).__name__}: {e}</span>"
            )
            result_text.value = ""

    def update_plateau_shading(*_):
        result_text.value = ""

        xs = state["xs"]
        if xs is None:
            return

        i0 = int(plateau_start.value)
        i1 = int(plateau_end.value)

        if i1 < i0:
            with fig_xi.batch_update():
                _clear_plateau_rectangle()
                fig_xi.data[1].x = []
                fig_xi.data[1].y = []
            return

        _set_plateau_visuals(show_mean_line=True)

    def extract_plateau(_):
        xs = state["xs"]
        xis = state["xis"]
        xis_u = state["xis_u"]

        if xs is None or len(xs) == 0:
            result_text.value = "<span style='color:red'>No ξ data available.</span>"
            return

        i0 = int(plateau_start.value)
        i1 = int(plateau_end.value)

        if i1 < i0:
            result_text.value = "<span style='color:red'>plateau j must be >= plateau i.</span>"
            return

        mask = (xs >= i0) & (xs <= i1)
        if not np.any(mask):
            result_text.value = "<span style='color:red'>Chosen plateau window contains no fitted points.</span>"
            return

        gv_xis = gv.gvar(xis, xis_u)
        plateau = gv_xis[mask]
        extracted_xi = np.mean(plateau)
        result_state["xi"] = extracted_xi

        with fig_xi.batch_update():
            fig_xi.data[1].x = [xs[0], xs[-1]]
            fig_xi.data[1].y = [extracted_xi.mean, extracted_xi.mean]

        result_text.value = (
            f"<b>plateau window</b>: [{i0}, {i1}] "
            f"&nbsp;&nbsp; <b>ξ</b>: {extracted_xi.mean:.6g} ± {extracted_xi.sdev:.3g}"
        )

    fit_dropdown.observe(update_plot, names="value")
    logcut_slider.observe(update_plot, names="value")
    plateau_start.observe(update_plateau_shading, names="value")
    plateau_end.observe(update_plateau_shading, names="value")
    extract_button.on_click(extract_plateau)

    update_plot()

    controls = widgets.VBox([
        widgets.HBox([fit_dropdown, logcut_slider]),
        widgets.HBox([plateau_start, plateau_end, extract_button]),
        status_text,
        result_text,
    ])

    display(controls)
    display(fig_xi)

    return {
        "fig_xi": fig_xi,
        "controls": controls,
        "fit_dropdown": fit_dropdown,
        "logcut_slider": logcut_slider,
        "plateau_start": plateau_start,
        "plateau_end": plateau_end,
        "extract_button": extract_button,
        "status_text": status_text,
        "result_text": result_text,
        "state": state,
        "result_state": result_state,
    }