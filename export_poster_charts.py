"""
export_poster_charts.py
Exports all poster figures as crisp 2x-scale PNGs using Plotly + kaleido.
Run from the project root: python3 export_poster_charts.py
Outputs land in ./poster_figures/
"""

import os, math, pathlib
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import stats

OUT = pathlib.Path("poster_figures")
OUT.mkdir(exist_ok=True)

CSV = "DebrisFlowVolume_Inventory.csv"

# ── Gartner engine ────────────────────────────────────────────────────────────
def gartner(i15, bmh_km2, relief_m):
    if bmh_km2 <= 0.001 or i15 <= 0 or relief_m <= 0:
        return 0.0
    try:
        return math.exp(
            4.22
            + 0.39 * math.sqrt(float(i15))
            + 0.36 * math.log(float(bmh_km2))
            + 0.13 * math.sqrt(float(relief_m))
        )
    except (ValueError, ZeroDivisionError):
        return 0.0

# ── Color scale ───────────────────────────────────────────────────────────────
RATIO_LO, RATIO_HI = 0.5, 2.0

def residual_color(ratio):
    def clamp(v, lo, hi): return max(lo, min(hi, v))
    if ratio <= 0:   return "#1d6fa4"
    if ratio < 1.0:
        t = clamp((1.0 - ratio) / (1.0 - RATIO_LO), 0, 1)
        r = int(245 - t * (245 - 29))
        g = int(240 - t * (240 - 111))
        b = int(232 - t * (232 - 164))
        return f"#{r:02x}{g:02x}{b:02x}"
    else:
        t = clamp((ratio - 1.0) / (RATIO_HI - 1.0), 0, 1)
        r = int(245 - t * (245 - 192))
        g = int(240 - t * (240 - 57))
        b = int(232 - t * (232 - 43))
        return f"#{r:02x}{g:02x}{b:02x}"

# ── Basin label helper ────────────────────────────────────────────────────────
_PREFIX = {"GRAND PRIX": "GP", "STATION": "ST", "THOMAS": "TH",
           "GRANDPRIX": "GP", "OLD": "OL"}

def basin_label(wid, fire):
    fire_up = fire.upper()
    for key, code in _PREFIX.items():
        wid_up = str(wid).upper()
        for variant in [key.replace(" ", ""), key, key.replace(" ", "_")]:
            if wid_up.startswith(variant):
                num = wid_up[len(variant):]
                return f"{code}-{num}"
    return str(wid)

# ── Load + aggregate data ─────────────────────────────────────────────────────
def load_fire(csv_path, fire_name):
    df = pd.read_csv(csv_path)
    df["FireName"] = df["FireName"].astype(str).str.strip().str.upper()
    df = df[df["FireName"] == fire_name.upper()].copy()
    df = df.dropna(subset=["Volume_m3", "AreaModHigh_km2", "Relief_m", "i15_mm/h"])
    df = df[(df["Volume_m3"] > 0) & (df["AreaModHigh_km2"] > 0.001)
            & (df["Relief_m"] > 0) & (df["i15_mm/h"] > 0)].copy()
    df["Predicted_m3"] = df.apply(
        lambda r: gartner(r["i15_mm/h"], r["AreaModHigh_km2"], r["Relief_m"]), axis=1
    )
    df = df[df["Predicted_m3"] > 0].copy()
    agg = df.groupby("WatershedID", as_index=False).agg(
        Volume_m3      =("Volume_m3",       "mean"),
        Predicted_m3   =("Predicted_m3",    "mean"),
        i15            =("i15_mm/h",        "mean"),
        AreaModHigh_km2=("AreaModHigh_km2", "mean"),
        Relief_m       =("Relief_m",        "mean"),
    )
    agg["Ratio"]  = agg["Predicted_m3"] / agg["Volume_m3"]
    agg["Label"]  = agg["WatershedID"].apply(lambda w: basin_label(w, fire_name))
    return agg.sort_values("Volume_m3", ascending=False).reset_index(drop=True)

# ── Plot theme ────────────────────────────────────────────────────────────────
PLOT_BG   = "rgba(13,27,42,1)"
PAPER_BG  = "rgba(0,0,0,0)"
GRID_COL  = "rgba(255,255,255,0.07)"
TEXT_COL  = "white"
FONT_FACE = "Calibri"

def base_layout(title, xtitle, ytitle, h=520):
    return dict(
        title=dict(text=title, font=dict(size=17, color=TEXT_COL, family=FONT_FACE), x=0.0, xanchor="left"),
        height=h,
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(color=TEXT_COL, size=13, family=FONT_FACE),
        xaxis=dict(title=xtitle, gridcolor=GRID_COL, color=TEXT_COL, tickfont=dict(size=11)),
        yaxis=dict(title=ytitle, gridcolor=GRID_COL, color=TEXT_COL, tickfont=dict(size=11)),
        margin=dict(t=80, b=80, l=70, r=20),
    )

# ── Save helper ───────────────────────────────────────────────────────────────
def save(fig, name, w=1400, h=520):
    path = OUT / f"{name}.png"
    fig.write_image(str(path), width=w, height=h, scale=2)
    print(f"  Saved {path}")

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Rank-preservation bar chart (one per fire)
# ═════════════════════════════════════════════════════════════════════════════
def make_rank_chart(df, fire_name):
    labels      = df["Label"].tolist()
    obs_vals    = df["Volume_m3"].tolist()
    pred_vals   = df["Predicted_m3"].tolist()
    ratios      = df["Ratio"].tolist()
    pred_colors = [residual_color(r) for r in ratios]

    rho_val, p_val = stats.spearmanr(pred_vals, obs_vals)
    rho_str = f"Spearman rho = {rho_val:.3f}  (p = {p_val:.3f})"

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Observed (USGS field)",
        x=labels, y=obs_vals,
        marker_color="rgba(180,180,180,0.85)",
        marker_line=dict(color="white", width=0.5),
        offsetgroup="obs",
        hovertemplate="<b>%{x}</b><br>Observed: %{y:,.0f} m3<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        name="Predicted (Gartner 2014)",
        x=labels, y=pred_vals,
        marker_color=pred_colors,
        marker_line=dict(color="white", width=0.5),
        offsetgroup="pred",
        hovertemplate="<b>%{x}</b><br>Predicted: %{y:,.0f} m3<extra></extra>",
    ))

    layout = base_layout(
        title=f"{fire_name.title()} -- Rank-Preservation Test",
        xtitle="Basin (sorted by observed volume, highest to lowest)",
        ytitle="Volume (m3, log scale)",
        h=520,
    )
    layout["yaxis"]["type"]   = "log"
    layout["barmode"]         = "group"
    layout["showlegend"]      = False
    layout["annotations"]     = [
        dict(x=0.01, y=-0.22, xref="paper", yref="paper",
             text=rho_str, showarrow=False,
             font=dict(size=12, color="#aaaaaa"), align="left"),
        dict(x=0.99, y=1.06, xref="paper", yref="paper",
             text="Gray = USGS observed  |  Colored = Gartner predicted  |  Blue=under  White=accurate  Red=over",
             showarrow=False, font=dict(size=10, color="#aaaaaa"), align="right"),
    ]
    fig.update_layout(**layout)
    return fig

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Ratio dot chart (one per fire)
# ═════════════════════════════════════════════════════════════════════════════
def make_ratio_chart(df, fire_name):
    labels  = df["Label"].tolist()
    ratios  = df["Ratio"].tolist()
    colors  = [residual_color(r) for r in ratios]
    ratio_labels = [f"{r:.2f}x" for r in ratios]

    fig = go.Figure()
    fig.add_hline(
        y=1.0, line_dash="dash",
        line_color="rgba(255,255,255,0.45)", line_width=1.5,
        annotation_text="perfect prediction (ratio = 1.0)",
        annotation_font_color="rgba(255,255,255,0.45)",
        annotation_position="right",
    )
    fig.add_trace(go.Scatter(
        x=labels, y=ratios,
        mode="markers+text",
        marker=dict(color=colors, size=18,
                    line=dict(color="white", width=1.2)),
        text=ratio_labels,
        textposition="top center",
        textfont=dict(size=11, color="white"),
        hovertemplate="<b>%{x}</b><br>Ratio: %{y:.2f}x<extra></extra>",
        showlegend=False,
    ))

    layout = base_layout(
        title=f"{fire_name.title()} -- Predicted / Observed Ratio per Basin",
        xtitle="Basin (sorted by observed volume, highest to lowest)",
        ytitle="Predicted / Observed ratio",
        h=400,
    )
    layout["yaxis"]["range"] = [0, max(max(ratios) * 1.35, 2.6)]
    layout["yaxis"]["zeroline"] = False
    fig.update_layout(**layout)
    return fig

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Model-wide scatter (all CA fires)
# ═════════════════════════════════════════════════════════════════════════════
def make_scatter(csv_path):
    df = pd.read_csv(csv_path)
    df["FireName"] = df["FireName"].astype(str).str.strip().str.upper()
    df = df.dropna(subset=["Volume_m3", "AreaModHigh_km2", "Relief_m", "i15_mm/h"])
    df = df[(df["Volume_m3"] > 0) & (df["AreaModHigh_km2"] > 0.001)
            & (df["Relief_m"] > 0) & (df["i15_mm/h"] > 0)].copy()
    if "State" in df.columns:
        df = df[df["State"].astype(str).str.strip().str.upper() == "CA"]
    df["Predicted_m3"] = df.apply(
        lambda r: gartner(r["i15_mm/h"], r["AreaModHigh_km2"], r["Relief_m"]), axis=1
    )
    df = df[df["Predicted_m3"] > 0].copy()
    df["Log_Obs"]  = np.log10(df["Volume_m3"].clip(lower=0.1))
    df["Log_Pred"] = np.log10(df["Predicted_m3"].clip(lower=0.1))

    r_val, _ = stats.pearsonr(df["Log_Obs"], df["Log_Pred"])
    sp_val, _ = stats.spearmanr(df["Log_Obs"], df["Log_Pred"])
    ratio = df["Predicted_m3"] / df["Volume_m3"]
    f2 = (ratio.between(0.5, 2.0).mean() * 100)

    all_vals = pd.concat([df["Volume_m3"], df["Predicted_m3"]])
    vmin = all_vals[all_vals > 0].min() * 0.3
    vmax = all_vals.max() * 3.0

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["Volume_m3"], y=df["Predicted_m3"],
        mode="markers",
        marker=dict(color="#e94560", size=7, opacity=0.82,
                    line=dict(color="white", width=0.3)),
        hovertemplate="Observed: %{x:,.0f} m3<br>Predicted: %{y:,.0f} m3<extra></extra>",
        showlegend=False,
    ))
    for y_vals, dash, opacity in [
        ([vmin, vmax],     "dash", 0.6),
        ([vmin*2, vmax*2], "dot",  0.25),
        ([vmin/2, vmax/2], "dot",  0.25),
    ]:
        fig.add_trace(go.Scatter(
            x=[vmin, vmax], y=y_vals, mode="lines",
            line=dict(dash=dash, color=f"rgba(255,255,255,{opacity})", width=1.5),
            showlegend=False, hoverinfo="skip",
        ))

    layout = base_layout(
        title=f"Model-Wide Accuracy -- {len(df)} USGS field measurements (California)",
        xtitle="Observed volume -- USGS field measured (m3)",
        ytitle="Predicted volume -- Gartner (2014) model (m3)",
        h=580,
    )
    layout["xaxis"]["type"] = "log"
    layout["yaxis"]["type"] = "log"
    layout["annotations"] = [dict(
        x=0.01, y=-0.17, xref="paper", yref="paper",
        text=f"R2 (log) = {r_val**2:.3f}   Spearman rho = {sp_val:.3f}   Within factor-of-2 = {f2:.0f}%   Dashed = 1:1 fit   Dotted = factor-of-2 tolerance",
        showarrow=False, font=dict(size=11, color="#aaaaaa"), align="left",
    )]
    fig.update_layout(**layout)
    return fig

# ═════════════════════════════════════════════════════════════════════════════
# RUN ALL EXPORTS
# ═════════════════════════════════════════════════════════════════════════════
print("Loading CSV...")
fires = ["GRAND PRIX", "STATION", "THOMAS"]
fire_codes = {"GRAND PRIX": "grandprix", "STATION": "station", "THOMAS": "thomas"}

for fire in fires:
    print(f"\n{fire}")
    df = load_fire(CSV, fire)
    print(f"  {len(df)} basins")

    fig_rank = make_rank_chart(df, fire)
    save(fig_rank, f"{fire_codes[fire]}_rank_chart", w=1400, h=520)

    fig_ratio = make_ratio_chart(df, fire)
    save(fig_ratio, f"{fire_codes[fire]}_ratio_chart", w=1200, h=400)

print("\nModel-wide scatter")
fig_scatter = make_scatter(CSV)
save(fig_scatter, "model_wide_scatter", w=1600, h=580)

print(f"\nAll figures saved to {OUT.resolve()}/")
