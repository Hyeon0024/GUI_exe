import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NaOH Lignin Removal Predictor",
    page_icon="🌲",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Constants: model performance (repeated random 80/20 split, n_repeats=5,
# random_state=42..46; mean +- std of Test R2/RMSE/MAE across repeats;
# source: logs/ml_naoh_remove_random_split_full_table_results.txt)
MODEL_PERFORMANCE = {
    "XGBoost": {
        "path": "models/tuned_xgboost.pkl",
        "test_r2": 0.9848, "test_r2_std": 0.0135,
        "test_rmse": 0.2984, "test_rmse_std": 0.0893,
        "test_mae": 0.2286, "test_mae_std": 0.0507,
        "params": "n_estimators=200, max_depth=3, lr=0.2",
        "color": "#e67e22",
    },
    "Extra Trees": {
        "path": "models/tuned_extra_trees.pkl",
        "test_r2": 0.9533, "test_r2_std": 0.0868,
        "test_rmse": 0.3911, "test_rmse_std": 0.4730,
        "test_mae": 0.2176, "test_mae_std": 0.1875,
        "params": "n_estimators=50, max_depth=None, min_samples_split=2",
        "color": "#2ecc71",
    },
    "SVR": {
        "path": "models/tuned_svr.pkl",
        "test_r2": 0.9239, "test_r2_std": 0.0647,
        "test_rmse": 0.6899, "test_rmse_std": 0.2972,
        "test_mae": 0.5560, "test_mae_std": 0.1332,
        "params": "C=100, epsilon=0.5, gamma=scale",
        "color": "#9b59b6",
    },
    "Random Forest": {
        "path": "models/tuned_random_forest.pkl",
        "test_r2": 0.9193, "test_r2_std": 0.0548,
        "test_rmse": 0.7281, "test_rmse_std": 0.2556,
        "test_mae": 0.4629, "test_mae_std": 0.1381,
        "params": "n_estimators=200, max_depth=None, min_samples_split=2",
        "color": "#3498db",
    },
    "Polynomial Ridge": {
        "path": "models/tuned_polynomial_ridge.pkl",
        "test_r2": 0.6548, "test_r2_std": 0.1272,
        "test_rmse": 1.5514, "test_rmse_std": 0.1243,
        "test_mae": 1.2215, "test_mae_std": 0.0977,
        "params": "degree=3, alpha=1.0",
        "color": "#e74c3c",
    },
}

# Training data statistics (from wood_lignin_NaOH.csv, n=36)
TRAIN_STATS = {
    "y_min": 10.315, "y_max": 24.362, "y_mean": 15.336, "y_std": 2.644,
    "ro_values": [4.25, 4.73, 4.95],
    "time_values": [12, 24],
    "conc_values": [0.5, 1.0, 2.0],
}

FEATURES = ['Severity Factor (Ro)', 'Bark_Binary', 'Pretreatment_Time (h)', 'Chemical_Concentration %)']

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')

# ── Model loader ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_all_models():
    loaded = {}
    for name, info in MODEL_PERFORMANCE.items():
        path = os.path.join(BASE_DIR, info["path"])
        try:
            loaded[name] = joblib.load(path)
        except Exception:
            loaded[name] = None
    return loaded

@st.cache_data
def load_training_data():
    path = os.path.join(BASE_DIR, 'data', 'wood_lignin_NaOH.csv')
    return pd.read_csv(path)

# ── Helper: build input DataFrame ─────────────────────────────────────────────
def make_input_df(ro, bark, time_h, conc):
    return pd.DataFrame({
        'Severity Factor (Ro)': [ro],
        'Bark_Binary': [bark],
        'Pretreatment_Time (h)': [time_h],
        'Chemical_Concentration %)': [conc],
    })

def is_in_training_range(ro, time_h, conc):
    ro_ok   = ro   in TRAIN_STATS["ro_values"]
    time_ok = time_h in TRAIN_STATS["time_values"]
    conc_ok = conc  in TRAIN_STATS["conc_values"]
    return ro_ok, time_ok, conc_ok

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .metric-card {
    background: #f4f6f9; border-radius: 10px; padding: 14px 18px;
    border-left: 5px solid #2ecc71; margin-bottom: 10px;
  }
  .metric-card.warn { border-left-color: #e67e22; }
  .section-title { font-size: 1.1rem; font-weight: 700; color: #2c3e50; margin-bottom: 6px; }
  .best-badge {
    background: #2ecc71; color: white; border-radius: 4px;
    padding: 1px 6px; font-size: 0.72rem; font-weight: bold;
  }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("NaOH Pretreatment — Lignin Removal Predictor")
st.markdown(
    "Machine-learning prediction of **lignin removal (%)** from lignocellulosic biomass "
    "under NaOH alkaline pretreatment. All models evaluated on a **held-out test set** "
    "(random 80/20 split, *n* = 108 samples from 36 experimental conditions; "
    "metrics are mean ± std over 5 repeated splits, random_state=42–46)."
)
st.divider()

# ── Load resources ────────────────────────────────────────────────────────────
models  = load_all_models()
df_data = load_training_data()

# ══════════════════════════════════════════════════════════════════════════════
# ROW 1 — Inputs | Model Selection | Prediction Result
# ══════════════════════════════════════════════════════════════════════════════
col_input, col_model, col_result = st.columns([1.4, 1.2, 1.4])

# ── LEFT: Input parameters ────────────────────────────────────────────────────
with col_input:
    st.markdown('<div class="section-title">Input Parameters</div>', unsafe_allow_html=True)

    ro = st.select_slider(
        "Severity Factor (R₀)",
        options=[4.25, 4.73, 4.95],
        value=4.73,
        help="Log₁₀ of the severity factor. Training data: 4.25, 4.73, 4.95",
    )
    bark = st.radio(
        "Bark inclusion",
        options=["Without bark (0)", "With bark (1)"],
        horizontal=True,
        help="Whether the biomass sample includes bark material",
    )
    bark_val = 1 if "With" in bark else 0

    time_h = st.select_slider(
        "Pretreatment Time (h)",
        options=[12, 24],
        value=12,
        help="Duration of NaOH pretreatment. Training data: 12 h, 24 h",
    )
    conc = st.select_slider(
        "NaOH Concentration (%)",
        options=[0.5, 1.0, 2.0],
        value=1.0,
        help="NaOH solution concentration (w/v %). Training data: 0.5, 1.0, 2.0",
    )

    # Extrapolation warning
    ro_ok, time_ok, conc_ok = is_in_training_range(ro, time_h, conc)
    if all([ro_ok, time_ok, conc_ok]):
        st.success("Input within training data range", icon="✅")
    else:
        out = [f"R₀={ro}" if not ro_ok else "",
               f"Time={time_h} h" if not time_ok else "",
               f"Conc={conc}%" if not conc_ok else ""]
        st.warning(f"Extrapolation — outside training range: {', '.join(x for x in out if x)}", icon="⚠️")

    st.markdown("**Input summary**")
    st.dataframe(
        pd.DataFrame({
            "Feature": ["Severity Factor (R₀)", "Bark", "Time (h)", "NaOH Conc (%)"],
            "Value": [ro, bark_val, time_h, conc],
            "Training Range": ["4.25 – 4.95", "0 / 1", "12 – 24", "0.5 – 2.0"],
        }),
        use_container_width=True, hide_index=True,
    )

# ── MIDDLE: Model selection ────────────────────────────────────────────────────
with col_model:
    st.markdown('<div class="section-title">Model Selection</div>', unsafe_allow_html=True)

    model_name = st.radio(
        "Active model",
        options=list(MODEL_PERFORMANCE.keys()),
        index=0,
        help="Select prediction model. Ranked by test set R².",
    )

    info = MODEL_PERFORMANCE[model_name]
    test_r2 = info["test_r2"]

    badge = '<span class="best-badge">BEST</span>' if model_name == "XGBoost" else ""
    card_cls = "metric-card" if test_r2 >= 0 else "metric-card warn"
    st.markdown(f"""
    <div class="{card_cls}">
      <b>{model_name}</b> {badge}<br>
      <small style="color:#7f8c8d">{info['params']}</small><br><br>
      <b>Test set performance</b> (random 80/20 split, mean ± std over 5 repeats)<br>
      R² = <b>{test_r2:.4f} ± {info['test_r2_std']:.4f}</b><br>
      RMSE = <b>{info['test_rmse']:.4f} ± {info['test_rmse_std']:.4f}%</b>&emsp;MAE = <b>{info['test_mae']:.4f} ± {info['test_mae_std']:.4f}%</b>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**All models — Test set R² (mean ± std, 5 repeats)**")
    perf_rows = sorted(
        [{"Model": nm, "Test R²": f"{d['test_r2']:.4f} ± {d['test_r2_std']:.4f}", "_sort": d["test_r2"]}
         for nm, d in MODEL_PERFORMANCE.items()],
        key=lambda x: x["_sort"], reverse=True,
    )
    perf_df = pd.DataFrame(perf_rows).drop(columns="_sort")
    st.dataframe(perf_df, use_container_width=True, hide_index=True)

# ── RIGHT: Prediction result ───────────────────────────────────────────────────
with col_result:
    st.markdown('<div class="section-title">Prediction Result</div>', unsafe_allow_html=True)

    input_df = make_input_df(ro, bark_val, time_h, conc)
    pipeline = models[model_name]

    if pipeline is not None:
        prediction = pipeline.predict(input_df)[0]

        # Gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prediction,
            number={"suffix": " %", "font": {"size": 36}},
            title={"text": "Predicted Lignin Removal", "font": {"size": 14}},
            gauge={
                "axis": {"range": [
                    TRAIN_STATS["y_min"] - 2,
                    TRAIN_STATS["y_max"] + 2,
                ], "ticksuffix": "%"},
                "bar": {"color": info["color"]},
                "steps": [
                    {"range": [TRAIN_STATS["y_min"] - 2, TRAIN_STATS["y_mean"] - TRAIN_STATS["y_std"]], "color": "#fde8e4"},
                    {"range": [TRAIN_STATS["y_mean"] - TRAIN_STATS["y_std"], TRAIN_STATS["y_mean"] + TRAIN_STATS["y_std"]], "color": "#e8f5e9"},
                    {"range": [TRAIN_STATS["y_mean"] + TRAIN_STATS["y_std"], TRAIN_STATS["y_max"] + 2], "color": "#e3f2fd"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 2},
                    "thickness": 0.75,
                    "value": TRAIN_STATS["y_mean"],
                },
            },
        ))
        fig_gauge.update_layout(height=230, margin=dict(t=30, b=0, l=20, r=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Contextual metrics
        z_score = (prediction - TRAIN_STATS["y_mean"]) / TRAIN_STATS["y_std"]
        percentile_approx = float(np.mean(df_data['Lignin_Removeal (%)'] <= prediction) * 100)

        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted", f"{prediction:.2f}%")
        c2.metric("vs. Train mean", f"{prediction - TRAIN_STATS['y_mean']:+.2f}%")
        c3.metric("Percentile", f"{percentile_approx:.0f}th")

        st.caption(
            f"Training data: *n* = 36 conditions | "
            f"mean = {TRAIN_STATS['y_mean']:.2f}% ± {TRAIN_STATS['y_std']:.2f}% (SD) | "
            f"range [{TRAIN_STATS['y_min']:.2f}, {TRAIN_STATS['y_max']:.2f}]%"
        )
        test_rmse = info["test_rmse"]
        st.caption(
            f"Model uncertainty (Test RMSE = ±{test_rmse:.4f}%) — "
            f"estimated 95% interval: **[{prediction - 2*test_rmse:.2f}, {prediction + 2*test_rmse:.2f}]%**"
        )
    else:
        st.error("Model file not found.")

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# ROW 2 — Analysis tabs
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs([
    "Sensitivity Analysis",
    "Multi-Model Comparison",
    "Training Data Distribution",
])

# ── TAB 1: Sensitivity analysis ───────────────────────────────────────────────
with tab1:
    st.markdown(
        "Predicted lignin removal as a single feature varies across its experimental range, "
        "with all other features held at current input values."
    )

    feat_label = st.selectbox(
        "Feature to sweep",
        options=["Severity Factor (R₀)", "Pretreatment Time (h)", "NaOH Concentration (%)"],
    )

    sweep_configs = {
        "Severity Factor (R₀)":    ("Severity Factor (Ro)",         np.linspace(4.25, 4.95, 60)),
        "Pretreatment Time (h)":   ("Pretreatment_Time (h)",        np.linspace(12, 24, 50)),
        "NaOH Concentration (%)":  ("Chemical_Concentration %)",    np.linspace(0.5, 2.0, 50)),
    }
    feat_col, sweep_vals = sweep_configs[feat_label]

    rows = []
    for v in sweep_vals:
        r  = v if feat_col == "Severity Factor (Ro)"      else ro
        t  = v if feat_col == "Pretreatment_Time (h)"     else time_h
        c  = v if feat_col == "Chemical_Concentration %)" else conc
        rows.append(make_input_df(r, bark_val, t, c))

    fig_sens = go.Figure()
    for nm, mdl in models.items():
        if mdl is None:
            continue
        preds = [mdl.predict(row)[0] for row in rows]
        lw = 3 if nm == model_name else 1.5
        dash = "solid" if nm == model_name else "dot"
        fig_sens.add_trace(go.Scatter(
            x=sweep_vals, y=preds, mode="lines",
            name=nm,
            line=dict(color=MODEL_PERFORMANCE[nm]["color"], width=lw, dash=dash),
        ))

    # Mark current input
    cur_x = {"Severity Factor (R₀)": ro,
              "Pretreatment Time (h)": time_h,
              "NaOH Concentration (%)": conc}[feat_label]
    cur_y = pipeline.predict(input_df)[0] if pipeline else None
    if cur_y is not None:
        fig_sens.add_trace(go.Scatter(
            x=[cur_x], y=[cur_y], mode="markers",
            marker=dict(symbol="diamond", size=14, color="black"),
            name="Current input",
        ))

    # Shaded training mean ± SD
    fig_sens.add_hrect(
        y0=TRAIN_STATS["y_mean"] - TRAIN_STATS["y_std"],
        y1=TRAIN_STATS["y_mean"] + TRAIN_STATS["y_std"],
        fillcolor="grey", opacity=0.1, line_width=0,
        annotation_text="Train mean ± SD",
        annotation_position="top left",
    )

    # Mark experimental levels
    exp_levels = {
        "Severity Factor (R₀)": TRAIN_STATS["ro_values"],
        "Pretreatment Time (h)": TRAIN_STATS["time_values"],
        "NaOH Concentration (%)": TRAIN_STATS["conc_values"],
    }[feat_label]
    for lv in exp_levels:
        fig_sens.add_vline(x=lv, line_dash="dash", line_color="grey", line_width=1)

    fig_sens.update_layout(
        xaxis_title=feat_label,
        yaxis_title="Predicted Lignin Removal (%)",
        legend_title="Model",
        height=400,
        margin=dict(t=20, b=40),
    )
    st.plotly_chart(fig_sens, use_container_width=True)
    st.caption("Dashed vertical lines: experimental levels used in training. Grey band: training data mean ± 1 SD.")

# ── TAB 2: Multi-model comparison ─────────────────────────────────────────────
with tab2:
    st.markdown(
        "Point predictions from all five models for the current input, "
        "overlaid with mean test set uncertainty (±1 RMSE, averaged over 5 repeated splits)."
    )

    pred_list = []
    for nm, mdl in models.items():
        if mdl is None:
            continue
        p = mdl.predict(input_df)[0]
        rmse_m = MODEL_PERFORMANCE[nm]["test_rmse"]
        pred_list.append({
            "Model": nm,
            "Prediction (%)": p,
            "Lower (−RMSE)": p - rmse_m,
            "Upper (+RMSE)": p + rmse_m,
            "Test R²": MODEL_PERFORMANCE[nm]["test_r2"],
            "Test RMSE": rmse_m,
        })
    pred_df = pd.DataFrame(pred_list).sort_values("Test R²", ascending=False)

    fig_multi = go.Figure()

    # Shaded region: training mean ± SD
    fig_multi.add_hrect(
        y0=TRAIN_STATS["y_mean"] - TRAIN_STATS["y_std"],
        y1=TRAIN_STATS["y_mean"] + TRAIN_STATS["y_std"],
        fillcolor="grey", opacity=0.1, line_width=0,
    )
    fig_multi.add_hline(
        y=TRAIN_STATS["y_mean"], line_dash="dash",
        line_color="grey", annotation_text="Train mean",
        annotation_position="right",
    )

    for _, row in pred_df.iterrows():
        nm = row["Model"]
        clr = MODEL_PERFORMANCE[nm]["color"]
        # Error bar = RMSE
        fig_multi.add_trace(go.Scatter(
            x=[row["Prediction (%)"]],
            y=[nm],
            mode="markers",
            marker=dict(size=14, color=clr,
                        line=dict(color="black", width=1.5)),
            error_x=dict(
                type="data",
                array=[row["Test RMSE"]],
                arrayminus=[row["Test RMSE"]],
                color=clr, thickness=2, width=6,
            ),
            name=nm,
            showlegend=False,
        ))

    fig_multi.update_layout(
        xaxis_title="Predicted Lignin Removal (%)",
        yaxis_title="",
        height=300,
        margin=dict(t=20, b=40, l=150),
    )
    st.plotly_chart(fig_multi, use_container_width=True)
    st.caption("Error bars: ±1 mean Test RMSE (random 80/20 split, *n* = 108, averaged over 5 repeats). Models ranked by Test R².")

    # Numeric table
    show_df = pred_df[["Model", "Prediction (%)", "Test R²", "Test RMSE"]].copy()
    show_df["Prediction (%)"] = show_df["Prediction (%)"].map("{:.3f}".format)
    show_df["Test R²"]        = show_df["Test R²"].map("{:.4f}".format)
    show_df["Test RMSE"]      = show_df["Test RMSE"].map("{:.4f}".format)
    st.dataframe(show_df, use_container_width=True, hide_index=True)

# ── TAB 3: Training data distribution ─────────────────────────────────────────
with tab3:
    st.markdown(
        "Distribution of observed lignin removal in the training dataset (*n* = 36 unique conditions), "
        "with the current model prediction overlaid."
    )

    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(
        x=df_data['Lignin_Removeal (%)'],
        nbinsx=12,
        name="Observed data",
        marker_color="#bdc3c7",
        opacity=0.8,
    ))

    if pipeline:
        pred_val = pipeline.predict(input_df)[0]
        fig_dist.add_vline(
            x=pred_val,
            line_color=info["color"], line_width=3,
            annotation_text=f"Prediction: {pred_val:.2f}%",
            annotation_position="top right",
            annotation_font_color=info["color"],
        )
        fig_dist.add_vline(
            x=TRAIN_STATS["y_mean"],
            line_dash="dash", line_color="grey", line_width=1.5,
            annotation_text=f"Mean: {TRAIN_STATS['y_mean']:.2f}%",
            annotation_position="top left",
        )

    # Group by experimental condition to add box plot
    fig_dist.update_layout(
        xaxis_title="Lignin Removal (%)",
        yaxis_title="Count",
        bargap=0.05,
        height=350,
        margin=dict(t=20, b=40),
    )
    st.plotly_chart(fig_dist, use_container_width=True)

    # Descriptive statistics
    desc = df_data['Lignin_Removeal (%)'].describe()
    stat_df = pd.DataFrame({
        "Statistic": ["n", "Mean", "SD", "Min", "Q1", "Median", "Q3", "Max"],
        "Value (%)": [
            f"{int(desc['count'])}",
            f"{desc['mean']:.3f}",
            f"{desc['std']:.3f}",
            f"{desc['min']:.3f}",
            f"{desc['25%']:.3f}",
            f"{desc['50%']:.3f}",
            f"{desc['75%']:.3f}",
            f"{desc['max']:.3f}",
        ],
    })
    st.dataframe(stat_df, use_container_width=True, hide_index=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Model training: random 80/20 train-test split (*n* = 108 samples, 36 conditions × 3 replicates), "
    "repeated 5× (random_state=42–46), fixed hyperparameters. "
    "Feature set: Severity Factor (R₀), Bark inclusion, Pretreatment Time, NaOH Concentration. "
    "Best model: XGBoost (Test R² = 0.9848 ± 0.0135, RMSE = 0.2984 ± 0.0893%)."
)
