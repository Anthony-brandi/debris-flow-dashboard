#!/usr/bin/env python3
"""
bootstrap_poster.py
Run this once from your project root to write poster_v2.js
to the current directory. Then run: node poster_v2.js

Usage:
    cd ~/Desktop/All\ Da\ Folders/Debris_Flow_App
    python3 bootstrap_poster.py
"""

import pathlib

HERE = pathlib.Path(__file__).parent

# ── poster_v2.js ─────────────────────────────────────────────────────────────
POSTER_JS = r"""const pptxgen = require("pptxgenjs");
const fs = require("fs");
const path = require("path");

const pres = new pptxgen();
pres.defineLayout({ name: "POSTER_48x36", width: 48, height: 36 });
pres.layout = "POSTER_48x36";

const slide = pres.addSlide();
slide.background = { color: "FFFFFF" };

const GREEN_DARK  = "144734";
const GREEN_MID   = "1A6349";
const GREEN_LIGHT = "D6E8DF";
const WHITE       = "FFFFFF";
const TEXT_DARK   = "1A1A1A";
const TEXT_MID    = "3A3A3A";
const TEXT_LITE   = "666666";
const ORANGE      = "C0392B";
const TEAL_ACC    = "0F7173";
const RULE        = "CCCCCC";

const HDR_H    = 5.4;
const BODY_Y   = HDR_H + 0.25;
const BODY_H   = 36 - BODY_Y - 0.35;
const MARGIN   = 0.55;
const COL_GAP  = 0.3;
const LEFT_W   = 10.2;
const RIGHT_W  = 10.2;
const CTR_W    = 48 - MARGIN*2 - LEFT_W - RIGHT_W - COL_GAP*2;
const LEFT_X   = MARGIN;
const CTR_X    = LEFT_X + LEFT_W + COL_GAP;
const RIGHT_X  = CTR_X + CTR_W + COL_GAP;

const mkS = () => ({ type: "outer", blur: 6, offset: 2, angle: 135, color: "000000", opacity: 0.10 });

function card(x, y, w, h, fillColor) {
  slide.addShape(pres.shapes.RECTANGLE, {
    x, y, w, h,
    fill: { color: fillColor || WHITE },
    line: { color: RULE, width: 0.75 },
    shadow: mkS(),
  });
}

function sectionHeader(x, y, w, text, light) {
  slide.addShape(pres.shapes.RECTANGLE, {
    x, y, w, h: 0.55,
    fill: { color: light ? GREEN_LIGHT : GREEN_DARK },
    line: { color: light ? GREEN_LIGHT : GREEN_DARK, width: 0 },
  });
  slide.addText(text, {
    x: x + 0.18, y: y + 0.07, w: w - 0.36, h: 0.4,
    fontSize: 20, bold: true,
    color: light ? GREEN_DARK : WHITE,
    fontFace: "Calibri", align: "left", margin: 0,
  });
}

function figPlaceholder(x, y, w, h, mainLabel, subLabel) {
  slide.addShape(pres.shapes.RECTANGLE, {
    x, y, w, h,
    fill: { color: "EEF4F0" },
    line: { color: GREEN_MID, width: 1.2, dashType: "dash" },
  });
  slide.addText(mainLabel, {
    x: x + 0.1, y: y + h/2 - 0.38, w: w - 0.2, h: 0.5,
    fontSize: 12, bold: true, color: GREEN_MID,
    align: "center", margin: 0, fontFace: "Calibri",
  });
  if (subLabel) {
    slide.addText(subLabel, {
      x: x + 0.1, y: y + h/2 + 0.15, w: w - 0.2, h: 0.38,
      fontSize: 10, italic: true, color: TEXT_LITE,
      align: "center", margin: 0, fontFace: "Calibri",
    });
  }
}

function tryAddImage(x, y, w, h, filename) {
  const fpath = path.join(__dirname, "poster_figures", filename);
  if (fs.existsSync(fpath)) {
    slide.addImage({ path: fpath, x, y, w, h, sizing: { type: "contain", w, h } });
    return true;
  }
  return false;
}

function figOrPlaceholder(x, y, w, h, filename, mainLabel, subLabel) {
  if (!tryAddImage(x, y, w, h, filename)) {
    figPlaceholder(x, y, w, h, mainLabel, subLabel);
  }
}

function caption(x, y, w, text) {
  slide.addText(text, {
    x, y, w, h: 0.65,
    fontSize: 11, color: TEXT_LITE, italic: true,
    align: "left", margin: 0, wrap: true, fontFace: "Calibri",
  });
}

function badge(x, y, w, rho, n, ratio, bgColor) {
  slide.addShape(pres.shapes.RECTANGLE, {
    x, y, w, h: 0.55,
    fill: { color: bgColor }, line: { color: bgColor, width: 0 },
  });
  slide.addText(
    `Spearman rho = ${rho}   |   n = ${n} basins   |   Mean ratio = ${ratio}`,
    { x: x+0.15, y: y+0.09, w: w-0.3, h: 0.37,
      fontSize: 13, bold: true, color: WHITE,
      align: "center", margin: 0, fontFace: "Calibri" }
  );
}

// HEADER
slide.addShape(pres.shapes.RECTANGLE, {
  x: 0, y: 0, w: 48, h: HDR_H,
  fill: { color: GREEN_DARK }, line: { color: GREEN_DARK, width: 0 },
});
slide.addText("Post-Fire Watershed Risk Portal", {
  x: 0.7, y: 0.5, w: 36, h: 2.2,
  fontSize: 72, bold: true, color: WHITE,
  fontFace: "Calibri", align: "left", margin: 0,
});
slide.addText(
  "Satellite-Driven Debris Flow Triage for Emergency Response  |  Integrating Google Earth Engine with USGS Empirical Volume Modeling",
  { x: 0.7, y: 2.85, w: 36, h: 0.9,
    fontSize: 20, color: "A8D5BE", italic: true,
    fontFace: "Calibri", align: "left", margin: 0 }
);
slide.addText(
  "Anthony Brandi   |   California Polytechnic State University, San Luis Obispo   |   CAFES Symposium, May 13, 2026",
  { x: 0.7, y: 3.9, w: 36, h: 0.65,
    fontSize: 17, color: "82BDA0",
    fontFace: "Calibri", align: "left", margin: 0 }
);
slide.addShape(pres.shapes.RECTANGLE, {
  x: 37.5, y: 1.55, w: 9.5, h: 2.4,
  fill: { color: "1A5C3A" }, line: { color: "82BDA0", width: 1 },
});
slide.addText("Cal Poly Logo\n+ NRES Department", {
  x: 37.5, y: 1.55, w: 9.5, h: 2.4,
  fontSize: 14, color: "A8D5BE", align: "center", valign: "middle", margin: 0,
});
slide.addShape(pres.shapes.RECTANGLE, {
  x: 44.6, y: 0.3, w: 3.0, h: 1.15,
  fill: { color: "1A5C3A" }, line: { color: "82BDA0", width: 1 },
});
slide.addText("QR: Live App", {
  x: 44.6, y: 0.3, w: 3.0, h: 1.15,
  fontSize: 11, color: "A8D5BE", align: "center", valign: "middle", margin: 0,
});

// LEFT COLUMN
let ly = BODY_Y;

card(LEFT_X, ly, LEFT_W, 8.2);
sectionHeader(LEFT_X, ly, LEFT_W, "Abstract");
slide.addText(
  [
    { text: "Post-fire debris flows are the leading secondary cause of wildfire fatalities in the western US. Current assessment methods require days of manual BAER field work — time that does not exist when atmospheric rivers follow within weeks of containment.\n\n", options: {} },
    { text: "The 2018 Thomas Fire / Montecito debris flow: 23 deaths, 5 weeks post-containment.\n\n", options: { bold: true, color: ORANGE } },
    { text: "This tool integrates Google Earth Engine satellite burn severity analysis with the USGS Gartner (2014) empirical volume model to rank-order watershed vulnerability within one hour of fire perimeter availability.", options: {} },
  ],
  { x: LEFT_X+0.25, y: ly+0.72, w: LEFT_W-0.5, h: 7.1,
    fontSize: 14, color: TEXT_DARK, align: "left", valign: "top",
    margin: 0, wrap: true, fontFace: "Calibri" }
);
ly += 8.5;

card(LEFT_X, ly, LEFT_W, 9.5);
sectionHeader(LEFT_X, ly, LEFT_W, "The Model");
slide.addShape(pres.shapes.RECTANGLE, {
  x: LEFT_X+0.25, y: ly+0.72, w: LEFT_W-0.5, h: 1.1,
  fill: { color: GREEN_LIGHT }, line: { color: GREEN_MID, width: 1 },
});
slide.addText("ln(V) = 4.22 + 0.39*sqrt(i15) + 0.36*ln(Bmh) + 0.13*sqrt(R)", {
  x: LEFT_X+0.35, y: ly+0.79, w: LEFT_W-0.7, h: 0.7,
  fontSize: 13, bold: true, color: GREEN_DARK,
  align: "center", fontFace: "Consolas", margin: 0,
});
slide.addText("Gartner et al. (2014)  |  Engineering Geology, 176, 45-56", {
  x: LEFT_X+0.25, y: ly+1.88, w: LEFT_W-0.5, h: 0.3,
  fontSize: 10, color: TEXT_LITE, italic: true, align: "center", margin: 0,
});
const varRows = [
  [
    { text: "Variable", options: { bold: true, color: WHITE, fill: { color: GREEN_DARK } } },
    { text: "Definition", options: { bold: true, color: WHITE, fill: { color: GREEN_DARK } } },
  ],
  [{ text: "i15" }, { text: "Peak 15-min rainfall intensity (mm/hr)" }],
  [{ text: "Bmh" }, { text: "Area burned at moderate-high severity (km2)" }],
  [{ text: "R" },   { text: "Watershed relief: max minus min elevation (m)" }],
];
slide.addTable(varRows, {
  x: LEFT_X+0.25, y: ly+2.25, w: LEFT_W-0.5, h: 2.1,
  fontSize: 12, color: TEXT_DARK,
  border: { pt: 0.75, color: RULE },
  fill: { color: WHITE }, colW: [1.5, 7.9],
  fontFace: "Calibri",
});
slide.addText(
  "Three satellite inputs. One risk ranking.\nDeliverable within one hour of containment.",
  { x: LEFT_X+0.25, y: ly+4.5, w: LEFT_W-0.5, h: 0.9,
    fontSize: 14, color: TEAL_ACC, bold: true, italic: true,
    align: "center", margin: 0, fontFace: "Calibri" }
);
slide.addText(
  "Data pipeline: USGS SRTM 30m DEM + ESA Sentinel-2 SR 10m burn severity + USGS WBD HUC-12 watersheds, processed live in Google Earth Engine Python API.",
  { x: LEFT_X+0.25, y: ly+5.6, w: LEFT_W-0.5, h: 1.2,
    fontSize: 12.5, color: TEXT_MID, align: "left", margin: 0, wrap: true, fontFace: "Calibri" }
);
ly += 9.8;

const opH = BODY_Y + BODY_H - ly;
card(LEFT_X, ly, LEFT_W, opH);
sectionHeader(LEFT_X, ly, LEFT_W, "Operational Output");
const opFigH = opH - 1.65;
figOrPlaceholder(
  LEFT_X+0.2, ly+0.65, LEFT_W-0.4, opFigH,
  "thomas_choropleth.png",
  "REPLACE: Thomas Fire choropleth map",
  "Module 2 at 91 mm/hr"
);
caption(
  LEFT_X+0.2, ly+0.65+opFigH+0.04, LEFT_W-0.4,
  "Fig. 1 -- HUC-12 basin predicted sediment yield, Thomas Fire (2017), recorded storm 91 mm/hr. Red = extreme risk."
);

// CENTER COLUMN
let cy = BODY_Y;
sectionHeader(CTR_X, cy, CTR_W, "Results — Validation Across Three Independent California Fires");
slide.addText(
  "Does the model rank basins correctly? Yes — Spearman rho > 0.89 across all three fires.",
  { x: CTR_X+0.2, y: cy+0.62, w: CTR_W-0.4, h: 0.48,
    fontSize: 16, color: TEAL_ACC, italic: true, bold: true,
    align: "center", margin: 0, fontFace: "Calibri" }
);
cy += 1.2;

const cFireH = (BODY_Y + BODY_H - cy - 1.2) / 3;

const fireData = [
  {
    name: "Grand Prix Fire (2003)",
    rho: "1.000", n: "7", ratio: "1.09x", badgeColor: "1A4731",
    note: "Best case: Perfect rank ordering. All 7 basins in correct risk sequence. Mean ratio 1.09x -- essentially unbiased.",
    rankFile: "grandprix_rank_chart.png", ratioFile: "grandprix_ratio_chart.png",
    figA: "Fig. 2a", figB: "Fig. 2b",
  },
  {
    name: "Station Fire (2009)",
    rho: "0.895", n: "20", ratio: "0.96x", badgeColor: "1E3A5F",
    note: "Strong case: Largest sample (20 basins). Rank correlation p < 0.001. Mean ratio 0.96x -- unbiased once repeat events averaged per basin.",
    rankFile: "station_rank_chart.png", ratioFile: "station_ratio_chart.png",
    figA: "Fig. 3a", figB: "Fig. 3b",
  },
  {
    name: "Thomas Fire (2018)",
    rho: "0.900", n: "5", ratio: "1.71x at 24 mm/hr", badgeColor: "7C2D12",
    note: "Boundary case: Rank ordering correct (p = 0.037). Over-prediction at 24 mm/hr explained by storm input mismatch, not model failure.",
    rankFile: "thomas_rank_chart.png", ratioFile: "thomas_ratio_chart.png",
    figA: "Fig. 4a", figB: "Fig. 4b",
  },
];

fireData.forEach((fire, i) => {
  const fy = cy + i * cFireH;
  const innerH = cFireH - 0.15;
  card(CTR_X, fy, CTR_W, innerH);
  slide.addShape(pres.shapes.RECTANGLE, {
    x: CTR_X, y: fy, w: CTR_W, h: 0.52,
    fill: { color: fire.badgeColor }, line: { color: fire.badgeColor, width: 0 },
  });
  slide.addText(fire.name, {
    x: CTR_X+0.2, y: fy+0.08, w: CTR_W-0.4, h: 0.36,
    fontSize: 16, bold: true, color: WHITE,
    align: "left", margin: 0, fontFace: "Calibri",
  });
  badge(CTR_X, fy+0.52, CTR_W, fire.rho, fire.n, fire.ratio, "1A3A4A");
  const chartY = fy + 1.13;
  const chartH = innerH - 2.0;
  const halfW  = (CTR_W - 0.35) / 2;
  figOrPlaceholder(CTR_X+0.15, chartY, halfW, chartH,
    fire.rankFile, `REPLACE: ${fire.name} rank chart`, "Sorted by observed volume");
  figOrPlaceholder(CTR_X+0.2+halfW, chartY, halfW, chartH,
    fire.ratioFile, `REPLACE: ${fire.name} ratio chart`, "Predicted/Observed ratio");
  caption(
    CTR_X+0.15, fy+innerH-0.75, CTR_W-0.3,
    `${fire.figA} -- Rank chart: gray = USGS observed, colored = Gartner predicted. ${fire.figB} -- Ratio dots: dashed line = perfect prediction. ${fire.note}`
  );
});

const sumY = BODY_Y + BODY_H - 1.05;
slide.addShape(pres.shapes.RECTANGLE, {
  x: CTR_X, y: sumY, w: CTR_W, h: 0.95,
  fill: { color: GREEN_LIGHT }, line: { color: GREEN_DARK, width: 1.2 },
});
slide.addText(
  "Rank ordering is preserved across all three fires (rho > 0.89). The model correctly identifies highest-risk basins for evacuation triage even when absolute volume predictions are imperfect.",
  { x: CTR_X+0.25, y: sumY+0.1, w: CTR_W-0.5, h: 0.75,
    fontSize: 14, color: GREEN_DARK, bold: true,
    align: "center", margin: 0, fontFace: "Calibri" }
);

// RIGHT COLUMN
let ry = BODY_Y;

const gaugeCardH = 9.2;
card(RIGHT_X, ry, RIGHT_W, gaugeCardH);
sectionHeader(RIGHT_X, ry, RIGHT_W, "The Interpolation Penalty", true);
slide.addText("Why does Thomas over-predict at 24 mm/hr?", {
  x: RIGHT_X+0.2, y: ry+0.65, w: RIGHT_W-0.4, h: 0.55,
  fontSize: 14, bold: true, color: TEXT_DARK,
  align: "left", margin: 0, fontFace: "Calibri", wrap: true,
});
figOrPlaceholder(RIGHT_X+0.2, ry+1.28, RIGHT_W-0.4, 6.3,
  "thomas_gauge_map.png", "REPLACE: Thomas gauge map",
  "Blue = gauges. Orange = basins.\nLine opacity = interpolation distance.");
caption(RIGHT_X+0.2, ry+7.66, RIGHT_W-0.4,
  "Fig. 5 -- Single ALERT gauge serving five Thomas Fire basins. Faded lines = high spatial interpolation of i15.");
ry += gaugeCardH + 0.2;

const stormCardH = 4.6;
card(RIGHT_X, ry, RIGHT_W, stormCardH);
sectionHeader(RIGHT_X, ry, RIGHT_W, "Storm Input Comparison", true);
const stormRows = [
  [
    { text: "Storm input", options: { bold: true, color: WHITE, fill: { color: GREEN_DARK } } },
    { text: "Mean ratio", options: { bold: true, color: WHITE, fill: { color: GREEN_DARK } } },
  ],
  [
    { text: "Design storm (24 mm/hr)" },
    { text: "1.71x  over-predicted", options: { color: ORANGE, bold: true } },
  ],
  [
    { text: "Recorded storm (91 mm/hr)" },
    { text: "~1.05x  near-accurate", options: { color: GREEN_DARK, bold: true } },
  ],
];
slide.addTable(stormRows, {
  x: RIGHT_X+0.2, y: ry+0.65, w: RIGHT_W-0.4, h: 2.3,
  fontSize: 13, color: TEXT_DARK,
  border: { pt: 0.75, color: RULE },
  fill: { color: WHITE }, colW: [6.0, 3.7],
  fontFace: "Calibri",
});
slide.addText("The model structure is correct. It needs accurate storm input.",
  { x: RIGHT_X+0.2, y: ry+3.1, w: RIGHT_W-0.4, h: 0.7,
    fontSize: 13, color: TEAL_ACC, italic: true, bold: true,
    align: "center", margin: 0, fontFace: "Calibri" });
ry += stormCardH + 0.2;

const tkH = BODY_Y + BODY_H - ry;
card(RIGHT_X, ry, RIGHT_W, tkH);
slide.addShape(pres.shapes.RECTANGLE, {
  x: RIGHT_X, y: ry, w: RIGHT_W, h: 0.55,
  fill: { color: GREEN_DARK }, line: { color: GREEN_DARK, width: 0 },
});
slide.addText("Key Takeaways", {
  x: RIGHT_X+0.2, y: ry+0.09, w: RIGHT_W-0.4, h: 0.38,
  fontSize: 20, bold: true, color: WHITE,
  fontFace: "Calibri", align: "left", margin: 0, charSpacing: 1,
});

const takeaways = [
  { head: "Triage accuracy holds.", body: "Spearman rho > 0.89 across three independent fires. The model correctly rank-orders basin vulnerability for evacuation prioritization even with imperfect rainfall data." },
  { head: "The interpolation penalty is real.", body: "Absolute volume error scales with gauge-to-basin distance. Rank ordering is robust. Volume magnitude accuracy requires co-located rain gauges." },
  { head: "Operational speed.", body: "Full basin-level risk assessment in under one hour of fire perimeter availability -- versus days for traditional BAER field assessment." },
  { head: "Future work.", body: "Sourcing ALERT gauge coordinates for Grand Prix (2003) and Station (2009) from San Bernardino Co. and LA County DPW will quantify the interpolation penalty empirically across all fires." },
];

let tkY = ry + 0.7;
takeaways.forEach((tk) => {
  slide.addShape(pres.shapes.RECTANGLE, {
    x: RIGHT_X+0.2, y: tkY+0.05, w: 0.07, h: 0.28,
    fill: { color: TEAL_ACC }, line: { color: TEAL_ACC, width: 0 },
  });
  slide.addText(
    [
      { text: tk.head + "  ", options: { bold: true, color: GREEN_DARK } },
      { text: tk.body, options: { color: TEXT_DARK } },
    ],
    { x: RIGHT_X+0.35, y: tkY, w: RIGHT_W-0.55, h: 2.2,
      fontSize: 13, align: "left", margin: 0, wrap: true, fontFace: "Calibri" }
  );
  tkY += 2.35;
});

// FOOTER
slide.addShape(pres.shapes.RECTANGLE, {
  x: 0, y: 35.65, w: 48, h: 0.35,
  fill: { color: GREEN_DARK }, line: { color: GREEN_DARK, width: 0 },
});
slide.addText(
  "Data: Crowder et al. (2025) USGS doi:10.5066/P13EZSWW  |  Model: Gartner et al. (2014) Engineering Geology 176:45-56  |  Satellite: ESA Sentinel-2 SR + USGS SRTM via Google Earth Engine",
  { x: 0.4, y: 35.66, w: 47, h: 0.3,
    fontSize: 9, color: "A8D5BE", align: "left", margin: 0, fontFace: "Calibri" }
);

[CTR_X - COL_GAP/2, RIGHT_X - COL_GAP/2].forEach(dx => {
  slide.addShape(pres.shapes.LINE, {
    x: dx, y: BODY_Y, w: 0, h: BODY_H,
    line: { color: RULE, width: 0.6 },
  });
});

pres.writeFile({ fileName: "PF-WRP_Poster_v2.pptx" })
  .then(() => console.log("Done: PF-WRP_Poster_v2.pptx"))
  .catch(e => { console.error(e); process.exit(1); });
"""

out = HERE / "poster_v2.js"
out.write_text(POSTER_JS)
print(f"Written: {out}")
print("\nNext steps:")
print("  pip install kaleido==0.2.1")
print("  python3 export_poster_charts.py")
print("  node poster_v2.js")
print("  open PF-WRP_Poster_v2.pptx")
