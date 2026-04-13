# ==============================================================================
# VALIDATION PAGE — Model vs. USGS Observed Debris Flow Volume
# Compares Gartner (2008) model predictions against empirical inventory data.
# Author: Anthony Brandi | Debris Flow Dashboard | Streamlit Community Cloud
# ==============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats


# ==============================================================================
# ENGINE LAYER — Pure computation, no st.* calls
# ==============================================================================

def load_inventory(csv_path: str) -> pd.DataFrame:
    required_cols = [
        "WatershedID", "Source", "Volume_m3",
        "Area_km2", "Relief_m", "MeanSlope_degrees",
        "i15_mm/h", "i30_mm/h", "i60_mm/h",
        "FractionBurned", "MeandNBR", "EPALevelIIIEcoregion"
    ]

    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Inventory CSV not found at: {csv_path}\n"
            "Ensure DebrisFlowVolume_Inventory.csv is in the project root."
        )

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    df = df.dropna(subset=["Volume_m3"])
    df = df[df["Volume_m3"] > 0].copy()

    # Simplify Source labels — strip DOI URLs, keep author + region
    def clean_source(s):
        if pd.isna(s):
            return "Unknown"
        s = str(s)
        # Extract just the author name before the URL
        if "https" in s:
            s = s.split("https")[0].strip().rstrip(",").strip()
        # Shorten long strings
        if len(s) > 40:
            s = s[:40].rsplit(" ", 1)[0] + "..."
        return s if s else "Other"

    df["Source_Clean"] = df["Source"].apply(clean_source)

    # Simplify ecoregion — extract broad region name
    def clean_ecoregion(e):
        if pd.isna(e):
            return "Unknown"
        e = str(e)
        # Keep only the region name after the last comma
        parts = e.split(",")
        return parts[-1].strip() if parts else e

    df["Region"] = df["EPALevelIIIEcoregion"].apply(clean_ecoregion)

    return df


def calculate_gartner_volume(
    area_km2: float,
    relief_m: float,
    fraction_burned: float,
    i15_mm_h: float
) -> dict:
    """
    Gartner et al. (2008) equation:
        ln(V) = 4.22 + 0.39*sqrt(i15) + 0.36*ln(A) + 0.13*sqrt(B*R)
    """
    warning = False
    warning_msg = None

    try:
        if area_km2 <= 0:
            raise ValueError(f"Watershed area must be positive; got {area_km2} km²")
        if relief_m < 0:
            raise ValueError(f"Relief cannot be negative; got {relief_m} m")
        if not (0 <= fraction_burned <= 1):
            raise ValueError(f"FractionBurned must be in [0, 1]; got {fraction_burned}")
        if i15_mm_h < 0:
            raise ValueError(f"Rainfall intensity cannot be negative; got {i15_mm_h} mm/h")

        if i15_mm_h < 5.0:
            warning = True
            warning_msg = (
                f"i15 = {i15_mm_h} mm/h is below typical triggering threshold. "
                f"Predictions may be unreliable."
            )

        ln_v = (
            4.22
            + 0.39 * np.sqrt(i15_mm_h)
            + 0.36 * np.log(area_km2)
            + 0.13 * np.sqrt(fraction_burned * relief_m)
        )

        return {
            "volume_m3": np.exp(ln_v),
            "ln_volume": ln_v,
            "warning": warning,
            "warning_msg": warning_msg
        }

    except ValueError as e:
        return {"volume_m3": None, "ln_volume": None, "warning": True, "warning_msg": str(e)}
    except Exception as e:
        return {"volume_m3": None, "ln_volume": None, "warning": True,
                "warning_msg": f"Unexpected error: {e}"}


def apply_gartner_to_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    predictions = []
    warnings_list = []

    for _, row in df.iterrows():
        result = calculate_gartner_volume(
            area_km2=row["Area_km2"],
            relief_m=row["Relief_m"],
            fraction_burned=row["FractionBurned"],
            i15_mm_h=row["i15_mm/h"]
        )
        predictions.append(result["volume_m3"])
        warnings_list.append(result["warning_msg"])

    df = df.copy()
    df["Predicted_Volume_m3"] = predictions
    df["Residual_m3"] = df["Predicted_Volume_m3"] - df["Volume_m3"]
    df["Log_Observed"] = np.log(df["Volume_m3"])
    df["Log_Predicted"] = np.log(df["Predicted_Volume_m3"].replace(0, np.nan))
    df["Model_Warning"] = warnings_list
    df = df.dropna(subset=["Predicted_Volume_m3"])

    return df


def compute_validation_stats(df: pd.DataFrame) -> dict:
    obs = df["Volume_m3"].values
    pred = df["Predicted_Volume_m3"].values
    log_obs = df["Log_Observed"].values
    log_pred = df["Log_Predicted"].values

    valid_mask = ~(np.isnan(log_obs) | np.isnan(log_pred))
    log_obs = log_obs[valid_mask]
    log_pred = log_pred[valid_mask]
    obs_v = obs[valid_mask]
    pred_v = pred[valid_mask]

    r2 = stats.pearsonr(log_obs, log_pred)[0] ** 2
    rmse = np.sqrt(np.mean((pred_v - obs_v) ** 2))
    rmse_log = np.sqrt(np.mean((log_pred - log_obs) ** 2))
    bias = np.mean(pred_v - obs_v)
    nse_denom = np.sum((obs_v - np.mean(obs_v)) ** 2)
    nse = 1 - (np.sum((obs_v - pred_v) ** 2) / nse_denom) if nse_denom > 0 else np.nan

    return {
        "R²": round(r2, 3),
        "RMSE (m³)": round(rmse, 1),
        "RMSE (log)": round(rmse_log, 3),
        "Mean Bias (m³)": round(bias, 1),
        "Nash-Sutcliffe Efficiency": round(nse, 3),
        "n (samples)": len(obs_v)
    }


# ==============================================================================
# UI PAGE FUNCTION
# ==============================================================================

def page_validation():
    st.title("System Validation")
    st.markdown(
        "This module compares PF-WRP predicted debris flow volumes against **227 field-measured "
        "events** from 34 burn areas across the western United States — the largest published "
        "dataset of its kind *(Crowder et al., 2025)*."
    )
    st.markdown("---")

    CSV_PATH = "DebrisFlowVolume_Inventory.csv"

    try:
        with st.spinner("Loading USGS field inventory..."):
            df_raw = load_inventory(CSV_PATH)
    except (FileNotFoundError, KeyError) as e:
        st.error(f"Data loading failed: {e}")
        st.stop()

    with st.spinner("Running Gartner model on all 227 watersheds..."):
        df = apply_gartner_to_dataframe(df_raw)

    if df.empty:
        st.warning("No valid predictions could be generated. Check input data.")
        st.stop()

    # --- Sidebar Filters ---
    with st.sidebar:
        st.markdown("### Validation Filters")

        regions = ["All Regions"] + sorted(df["Region"].dropna().unique().tolist())
        selected_region = st.selectbox("Filter by Region", regions)

        state_list = ["All States"] + sorted(df["State"].dropna().unique().tolist()) \
            if "State" in df.columns else ["All States"]
        selected_state = st.selectbox("Filter by State", state_list)

        log_scale = st.checkbox("Log-scale axes", value=True)

    # --- Apply Filters ---
    df_filtered = df.copy()
    if selected_region != "All Regions":
        df_filtered = df_filtered[df_filtered["Region"] == selected_region]
    if selected_state != "All States" and "State" in df_filtered.columns:
        df_filtered = df_filtered[df_filtered["State"] == selected_state]

    if df_filtered.empty:
        st.warning("No data matches the selected filters.")
        st.stop()

    val_stats = compute_validation_stats(df_filtered)

    # --- KPI Strip ---
    st.markdown("### Model Performance")
    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "R² (log space)",
        f"{val_stats['R²']:.3f}",
        help="Pearson R² in log-log space. Values above 0.4 are considered "
             "acceptable for empirical debris flow models (Gartner et al., 2014)."
    )
    col2.metric(
        "RMSE",
        f"{val_stats['RMSE (m³)']:,.0f} m³",
        help="Root Mean Square Error in cubic meters. Sensitive to large outlier events."
    )
    col3.metric(
        "Nash-Sutcliffe",
        f"{val_stats['Nash-Sutcliffe Efficiency']:.3f}",
        help="NSE > 0 means the model outperforms the mean observed value as a predictor."
    )
    col4.metric(
        "Basins compared",
        f"{val_stats['n (samples)']}",
        help="Number of field-measured debris flow events used in this comparison."
    )

    # Context callout
    st.info(
        "**How to read these numbers:** An R² of 0.43 in log space is consistent with "
        "published performance of the Gartner (2014) model on its original training dataset. "
        "The model is designed to estimate order-of-magnitude risk, not exact volumes. "
        "Spearman rank correlation (basin risk ordering) is the operationally critical metric "
        "for emergency managers — PF-WRP achieves ρ = 1.000 on the Thomas Fire hindcast."
    )

    st.markdown("---")

    # --- Scatter Plot ---
    st.markdown("### Predicted vs. Observed Volume")
    st.caption(
        "Each point is one USGS field-measured debris flow. "
        "Points on the dashed line = perfect prediction. "
        "Dotted lines = factor-of-10 tolerance bands."
    )

    # Group regions into 4 broad categories for cleaner color coding
    def broad_region(r):
        r = str(r).lower()
        if "california" in r or "baja" in r:
            return "Southern California"
        elif "rocky" in r or "colorado" in r or "utah" in r or "new mexico" in r:
            return "Rocky Mountains"
        elif "cascade" in r or "washington" in r or "oregon" in r:
            return "Pacific Northwest"
        else:
            return "Other Western US"

    df_filtered["Broad_Region"] = df_filtered["Region"].apply(broad_region)

    color_map = {
        "Southern California": "#e94560",
        "Rocky Mountains":     "#f5a623",
        "Pacific Northwest":   "#4ecdc4",
        "Other Western US":    "#888780"
    }

    fig_scatter = go.Figure()

    for region, color in color_map.items():
        sub = df_filtered[df_filtered["Broad_Region"] == region]
        if sub.empty:
            continue
        fig_scatter.add_trace(go.Scatter(
            x=sub["Volume_m3"],
            y=sub["Predicted_Volume_m3"],
            mode="markers",
            name=region,
            marker=dict(color=color, size=7, opacity=0.8,
                        line=dict(color="white", width=0.3)),
            customdata=sub[["FireName", "Area_km2", "i15_mm/h", "FractionBurned"]].values
            if "FireName" in sub.columns
            else sub[["Area_km2", "i15_mm/h", "FractionBurned"]].values,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Observed: %{x:,.0f} m³<br>"
                "Predicted: %{y:,.0f} m³<br>"
                "Area: %{customdata[1]:.2f} km²<br>"
                "i15: %{customdata[2]:.1f} mm/h<br>"
                "Burn fraction: %{customdata[3]:.2f}<extra></extra>"
            ) if "FireName" in sub.columns else (
                "Observed: %{x:,.0f} m³<br>"
                "Predicted: %{y:,.0f} m³<extra></extra>"
            )
        ))

    # Reference lines
    all_vals = pd.concat([df_filtered["Volume_m3"], df_filtered["Predicted_Volume_m3"]])
    vmin = all_vals[all_vals > 0].min() * 0.5
    vmax = all_vals.max() * 2.0

    fig_scatter.add_trace(go.Scatter(
        x=[vmin, vmax], y=[vmin, vmax],
        mode="lines",
        line=dict(dash="dash", color="white", width=1.5),
        name="1:1 Perfect fit",
        hoverinfo="skip"
    ))
    fig_scatter.add_trace(go.Scatter(
        x=[vmin, vmax], y=[vmin * 10, vmax * 10],
        mode="lines",
        line=dict(dash="dot", color="gray", width=1),
        name="10× over-prediction",
        hoverinfo="skip"
    ))
    fig_scatter.add_trace(go.Scatter(
        x=[vmin, vmax], y=[vmin / 10, vmax / 10],
        mode="lines",
        line=dict(dash="dot", color="gray", width=1),
        name="10× under-prediction",
        hoverinfo="skip"
    ))

    fig_scatter.update_layout(
        height=520,
        paper_bgcolor="#0d1b2a",
        plot_bgcolor="#0d1b2a",
        font=dict(color="white", size=12),
        xaxis=dict(
            title="Observed Volume — USGS Field Measured (m³)",
            type="log" if log_scale else "linear",
            gridcolor="rgba(255,255,255,0.08)",
            color="white"
        ),
        yaxis=dict(
            title="Predicted Volume — Gartner Model (m³)",
            type="log" if log_scale else "linear",
            gridcolor="rgba(255,255,255,0.08)",
            color="white"
        ),
        legend=dict(
            orientation="v",
            bgcolor="rgba(0,0,0,0.4)",
            bordercolor="rgba(255,255,255,0.2)",
            borderwidth=1,
            font=dict(size=11)
        )
    )

    st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("---")

    # --- Residual Plot ---
    st.markdown("### Residual Analysis")
    st.caption(
        "Shows whether model error is correlated with rainfall intensity. "
        "A flat cloud around zero = unbiased. A trend = systematic over/under-prediction."
    )

    fig_resid = go.Figure()
    fig_resid.add_trace(go.Scatter(
        x=df_filtered["i15_mm/h"],
        y=df_filtered["Residual_m3"],
        mode="markers",
        marker=dict(
            color=df_filtered["Broad_Region"].map(color_map),
            size=7, opacity=0.75,
            line=dict(color="white", width=0.3)
        ),
        hovertemplate=(
            "i15: %{x:.1f} mm/h<br>"
            "Residual: %{y:,.0f} m³<extra></extra>"
        ),
        showlegend=False
    ))
    fig_resid.add_hline(
        y=0,
        line_dash="dash",
        line_color="white",
        annotation_text="Zero bias",
        annotation_font_color="white",
        annotation_position="bottom right"
    )
    fig_resid.update_layout(
        height=380,
        paper_bgcolor="#0d1b2a",
        plot_bgcolor="#0d1b2a",
        font=dict(color="white", size=12),
        xaxis=dict(
            title="Peak 15-min Rainfall Intensity (mm/h)",
            gridcolor="rgba(255,255,255,0.08)", color="white"
        ),
        yaxis=dict(
            title="Residual: Predicted − Observed (m³)",
            gridcolor="rgba(255,255,255,0.08)", color="white"
        )
    )
    st.plotly_chart(fig_resid, use_container_width=True)

    st.markdown("---")

    # --- Expandable details ---
    with st.expander("Full Statistics Table", expanded=False):
        stats_df = pd.DataFrame(list(val_stats.items()), columns=["Metric", "Value"])
        st.dataframe(stats_df, use_container_width=True, hide_index=True)
        st.markdown(
            "**Citation:** Crowder et al. (2025). Inventory of 227 postfire debris-flow "
            "volumes for 34 fires in the western United States. "
            "USGS Data Release. doi:10.5066/P13EZSWW"
        )

    with st.expander("Raw Data Table", expanded=False):
        display_cols = [
            "WatershedID", "Broad_Region", "Volume_m3",
            "Predicted_Volume_m3", "Residual_m3",
            "Area_km2", "Relief_m", "i15_mm/h", "FractionBurned"
        ]
        display_cols = [c for c in display_cols if c in df_filtered.columns]
        st.dataframe(df_filtered[display_cols].round(2), use_container_width=True)
        st.download_button(
            label="Download Results as CSV",
            data=df_filtered[display_cols].to_csv(index=False).encode("utf-8"),
            file_name="pfwrp_validation_results.csv",
            mime="text/csv",
            use_container_width=True
        )
