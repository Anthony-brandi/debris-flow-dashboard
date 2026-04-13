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
    """
    Load and validate the USGS debris flow inventory CSV.

    Args:
        csv_path: Relative path to DebrisFlowVolume_Inventory.csv

    Returns:
        pd.DataFrame: Cleaned inventory with required columns verified.

    Raises:
        FileNotFoundError: If CSV does not exist at the specified path.
        KeyError: If required columns are missing from the dataset.
    """
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

    # Drop rows with null observed volume — cannot validate without ground truth
    df = df.dropna(subset=["Volume_m3"])

    # Remove physically impossible values (volume must be positive)
    df = df[df["Volume_m3"] > 0].copy()

    return df


def calculate_gartner_volume(
    area_km2: float,
    relief_m: float,
    fraction_burned: float,
    i15_mm_h: float
) -> dict:
    """
    Compute predicted debris flow volume using the Gartner et al. (2008) equation.

    Equation:
        ln(V) = 4.22 + 0.39*sqrt(i15) + 0.36*ln(A) + 0.13*sqrt(B*R)

    Where:
        V   = Volume (m³)
        i15 = Peak 15-min rainfall intensity (mm/h)
        A   = Watershed area (km²)
        B   = Fraction of watershed burned at moderate-or-high severity (0–1)
        R   = Local relief (m)

    Reference:
        Gartner, J.E., et al. (2008). A pragmatic approach for estimating
        debris flow volumes from rainfall. Geomorphology, 96(3-4), 121-139.

    Args:
        area_km2:        Watershed area in km²
        relief_m:        Local relief (max - min elevation) in meters
        fraction_burned: Fraction of basin burned at moderate+ severity (0–1)
        i15_mm_h:        Peak 15-minute rainfall intensity in mm/h

    Returns:
        dict with keys:
            'volume_m3'     : Predicted volume (float, m³)
            'ln_volume'     : Natural log of predicted volume (float)
            'warning'       : True if any input triggered a domain guard (bool)
            'warning_msg'   : Human-readable warning string or None
    """
    warning = False
    warning_msg = None

    try:
        # --- Input validation guards ---
        if area_km2 <= 0:
            raise ValueError(f"Watershed area must be positive; got {area_km2} km²")
        if relief_m < 0:
            raise ValueError(f"Relief cannot be negative; got {relief_m} m")
        if not (0 <= fraction_burned <= 1):
            raise ValueError(
                f"FractionBurned must be in [0, 1]; got {fraction_burned}"
            )
        if i15_mm_h < 0:
            raise ValueError(
                f"Rainfall intensity cannot be negative; got {i15_mm_h} mm/h"
            )

        # --- Warn on near-zero inputs that produce unreliable predictions ---
        if i15_mm_h < 5.0:
            warning = True
            warning_msg = (
                f"i15 = {i15_mm_h} mm/h is below typical triggering threshold "
                f"(~5 mm/h). Gartner model predictions may be unreliable."
            )

        # --- Gartner (2008) Equation ---
        ln_v = (
            4.22
            + 0.39 * np.sqrt(i15_mm_h)
            + 0.36 * np.log(area_km2)          # natural log
            + 0.13 * np.sqrt(fraction_burned * relief_m)
        )

        volume_m3 = np.exp(ln_v)

        return {
            "volume_m3": volume_m3,
            "ln_volume": ln_v,
            "warning": warning,
            "warning_msg": warning_msg
        }

    except ValueError as e:
        return {
            "volume_m3": None,
            "ln_volume": None,
            "warning": True,
            "warning_msg": str(e)
        }
    except Exception as e:
        return {
            "volume_m3": None,
            "ln_volume": None,
            "warning": True,
            "warning_msg": f"Unexpected error in Gartner calculation: {e}"
        }


def apply_gartner_to_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the Gartner (2008) model to every row in the inventory DataFrame.

    Uses i15_mm/h, Area_km2, Relief_m, and FractionBurned columns.
    Appends 'Predicted_Volume_m3' and 'Residual_m3' columns.

    Args:
        df: Cleaned inventory DataFrame from load_inventory()

    Returns:
        pd.DataFrame: Original df with model prediction columns appended.
    """
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

    # Drop rows where prediction failed (None) — flagged warnings are retained
    df = df.dropna(subset=["Predicted_Volume_m3"])

    return df


def compute_validation_stats(df: pd.DataFrame) -> dict:
    """
    Compute standard model validation statistics.

    Metrics returned:
        - R²         : Coefficient of determination (log-log space)
        - RMSE       : Root Mean Square Error (m³)
        - RMSE_log   : RMSE in log space (dimensionless — model skill metric)
        - Bias       : Mean residual (positive = over-prediction)
        - NSE        : Nash-Sutcliffe Efficiency (1.0 = perfect)
        - n          : Sample size

    Args:
        df: DataFrame with 'Volume_m3', 'Predicted_Volume_m3',
            'Log_Observed', 'Log_Predicted' columns.

    Returns:
        dict of validation metric name → float
    """
    obs = df["Volume_m3"].values
    pred = df["Predicted_Volume_m3"].values
    log_obs = df["Log_Observed"].values
    log_pred = df["Log_Predicted"].dropna().values

    # Align lengths after any NaN drop
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
# UI PAGE FUNCTION — Called from sidebar router
# ==============================================================================

def page_validation():
    """
    Renders the Model Validation page.
    Compares Gartner (2008) predictions against USGS inventory observations.
    Follows the SPA pattern: reads data, computes, then renders UI.
    """
    st.title("4. Model Validation")
    st.markdown(
        "Compares predicted debris flow volumes (Gartner et al., 2008) against "
        "empirical observations from the USGS post-fire debris flow inventory."
    )
    st.markdown("---")

    # --- Load Data ---
    CSV_PATH = "DebrisFlowVolume_Inventory.csv"

    try:
        with st.spinner("Loading USGS inventory..."):
            df_raw = load_inventory(CSV_PATH)
    except (FileNotFoundError, KeyError) as e:
        st.error(f"❌ Data loading failed: {e}")
        st.stop()

    # --- Apply Model ---
    with st.spinner("Applying Gartner (2008) model to all watersheds..."):
        df = apply_gartner_to_dataframe(df_raw)

    if df.empty:
        st.warning("⚠️ No valid predictions could be generated. Check input data.")
        st.stop()

    # --- Sidebar Filters ---
    with st.sidebar:
        st.markdown("### 🔽 Filter Validation Data")

        sources = ["All"] + sorted(df["Source"].dropna().unique().tolist())
        selected_source = st.selectbox("Data Source", sources)

        ecoregions = ["All"] + sorted(df["EPALevelIIIEcoregion"].dropna().unique().tolist())
        selected_eco = st.selectbox("EPA Ecoregion", ecoregions)

        log_scale = st.checkbox("Log-scale axes", value=True)
        show_warnings = st.checkbox("Highlight model warnings", value=True)

    # --- Apply Filters ---
    df_filtered = df.copy()
    if selected_source != "All":
        df_filtered = df_filtered[df_filtered["Source"] == selected_source]
    if selected_eco != "All":
        df_filtered = df_filtered[df_filtered["EPALevelIIIEcoregion"] == selected_eco]

    if df_filtered.empty:
        st.warning("⚠️ No data matches the selected filters.")
        st.stop()

    # --- Compute Stats on Filtered Data ---
    val_stats = compute_validation_stats(df_filtered)

    # --- KPI Metrics Row ---
    st.subheader("📊 Validation Statistics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("R² (log-log)", f"{val_stats['R²']:.3f}")
    col2.metric("RMSE", f"{val_stats['RMSE (m³)']:,.0f} m³")
    col3.metric("Nash-Sutcliffe", f"{val_stats['Nash-Sutcliffe Efficiency']:.3f}")
    col4.metric("Sample Size (n)", f"{val_stats['n (samples)']}")

    st.markdown("---")

    # --- Primary Scatter Plot: Observed vs. Predicted ---
    st.subheader("Observed vs. Predicted Volume")

    color_col = "Model_Warning" if show_warnings else "Source"
    hover_cols = [
        "WatershedID", "Source", "Area_km2", "Relief_m",
        "i15_mm/h", "FractionBurned", "MeandNBR", "Model_Warning"
    ]
    # Only include hover cols that exist in df_filtered
    hover_cols = [c for c in hover_cols if c in df_filtered.columns]

    fig_scatter = px.scatter(
        df_filtered,
        x="Volume_m3",
        y="Predicted_Volume_m3",
        color="Source",
        symbol="EPALevelIIIEcoregion" if "EPALevelIIIEcoregion" in df_filtered.columns else None,
        hover_data=hover_cols,
        labels={
            "Volume_m3": "Observed Volume (m³)",
            "Predicted_Volume_m3": "Gartner Predicted Volume (m³)",
            "Source": "Data Source"
        },
        title="Gartner (2008) Model Validation: Predicted vs. Observed Debris Flow Volume",
        template="plotly_white",
        log_x=log_scale,
        log_y=log_scale,
    )

    # 1:1 Perfect Agreement Line
    all_vals = pd.concat([df_filtered["Volume_m3"], df_filtered["Predicted_Volume_m3"]])
    vmin, vmax = all_vals.min() * 0.5, all_vals.max() * 2.0

    fig_scatter.add_trace(go.Scatter(
        x=[vmin, vmax],
        y=[vmin, vmax],
        mode="lines",
        line=dict(dash="dash", color="black", width=1.5),
        name="1:1 Line (Perfect Agreement)"
    ))

    # Optional: ±1 order of magnitude bands
    fig_scatter.add_trace(go.Scatter(
        x=[vmin, vmax],
        y=[vmin * 10, vmax * 10],
        mode="lines",
        line=dict(dash="dot", color="gray", width=1),
        name="+1 Order of Magnitude"
    ))
    fig_scatter.add_trace(go.Scatter(
        x=[vmin, vmax],
        y=[vmin / 10, vmax / 10],
        mode="lines",
        line=dict(dash="dot", color="gray", width=1),
        name="−1 Order of Magnitude"
    ))

    fig_scatter.update_layout(
        height=550,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_title="Observed Volume (m³)",
        yaxis_title="Gartner Predicted Volume (m³)",
        font=dict(family="Arial", size=12)
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.caption(
        "Dashed black line = 1:1 perfect agreement. Dotted lines = ±1 order of magnitude. "
        "Points above the 1:1 line indicate model over-prediction."
    )

    st.markdown("---")

    # --- Secondary Plot: Residuals vs. i15 intensity ---
    st.subheader("Residual Analysis: Bias vs. Rainfall Intensity")

    fig_resid = px.scatter(
        df_filtered,
        x="i15_mm/h",
        y="Residual_m3",
        color="Source",
        hover_data=["WatershedID", "Volume_m3", "Predicted_Volume_m3"],
        labels={
            "i15_mm/h": "Peak 15-min Intensity (mm/h)",
            "Residual_m3": "Residual: Predicted − Observed (m³)"
        },
        title="Residual Plot: Is Model Bias Correlated with Rainfall Intensity?",
        template="plotly_white"
    )

    # Zero-bias reference line
    fig_resid.add_hline(
        y=0,
        line_dash="dash",
        line_color="black",
        annotation_text="Zero Bias",
        annotation_position="bottom right"
    )

    fig_resid.update_layout(height=400)
    st.plotly_chart(fig_resid, use_container_width=True)

    st.caption(
        "Points above zero indicate over-prediction; below zero indicate under-prediction. "
        "A systematic trend suggests the model is sensitive to intensity in a nonlinear way."
    )

    st.markdown("---")

    # --- Full Stats Table ---
    with st.expander("📋 Full Validation Statistics Table", expanded=False):
        stats_df = pd.DataFrame(
            list(val_stats.items()),
            columns=["Metric", "Value"]
        )
        st.dataframe(stats_df, use_container_width=True, hide_index=True)

    # --- Raw Data Table ---
    with st.expander("📂 Filtered Inventory Data", expanded=False):
        display_cols = [
            "WatershedID", "Source", "EPALevelIIIEcoregion",
            "Volume_m3", "Predicted_Volume_m3", "Residual_m3",
            "Area_km2", "Relief_m", "i15_mm/h", "FractionBurned", "MeandNBR"
        ]
        display_cols = [c for c in display_cols if c in df_filtered.columns]
        st.dataframe(
            df_filtered[display_cols].round(2),
            use_container_width=True
        )
        st.download_button(
            label="⬇️ Download Filtered Results as CSV",
            data=df_filtered[display_cols].to_csv(index=False).encode("utf-8"),
            file_name="validation_results_filtered.csv",
            mime="text/csv"
        )
