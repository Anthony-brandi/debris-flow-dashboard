# CLAUDE.md — PF-WRP Project Instructions
# Post-Fire Watershed Risk Portal | Anthony Brandi | Cal Poly SLO
# This file governs all Claude Code behavior for this project.

---

## 🧠 PROJECT IDENTITY

**Project:** Post-Fire Watershed Risk Portal (PF-WRP)
**Student:** Anthony Brandi | Cal Poly SLO | CAFES Symposium May 13, 2026
**GitHub:** github.com/Anthony-brandi/debris-flow-dashboard
**App:** Streamlit + Google Earth Engine Python API

**Scientific Core:**
- Model: Gartner et al. (2014) USGS empirical logistic regression
- Formula: ln(V) = 4.22 + 0.13·ln(B23) + 0.36·ln(R15) + 0.39·√(HM)
- Validation: USGS Crowder et al. (2025) doi:10.5066/P13EZSWW

---

## 📁 PATH CONFIGURATION
# Update these paths to match your actual machine before first use.

OBSIDIAN_VAULT = "~/Documents/PF-WRP Research Hub"
PROJECT_ROOT   = "~/Debris_Flow_App"
SESSIONS_DIR   = "~/Documents/PF-WRP Research Hub/Sessions"
SCIENCE_DIR    = "~/Documents/PF-WRP Research Hub/Science"
CODE_DIR       = "~/Documents/PF-WRP Research Hub/Code"
ACADEMIC_DIR   = "~/Documents/PF-WRP Research Hub/Academic"
DATA_DIR       = "~/Documents/PF-WRP Research Hub/Data"
CONTEXT_NOTE   = "~/Documents/PF-WRP Research Hub/Sessions/Claude_Context_Template.md"
MOC_NOTE       = "~/Documents/PF-WRP Research Hub/🗺️ MOC — Map of Content.md"

---

## 📋 CODING DIRECTIVES
# These rules apply to ALL code generation in this project. No exceptions.

1. **Zero Shortcuts** — Write every line of every function explicitly.
2. **No Placeholders** — Never use `# ... rest of code ...`, `# insert here`, or `pass`.
3. **Production-Ready** — All output must be copy-paste ready into the live app.
4. **Complete Blocks** — When updating app.py, provide the entire updated block,
   not just the changed lines.
5. **Statistical Rigor** — When doing validation work, always include the full
   mathematical logic for RMSE, R², Spearman ρ, and percent error.
6. **GEE Constraints** — All Google Earth Engine calls must include:
   - `.simplify(maxError=100)` on geometry
   - `tileScale=16` on reduceRegions
   - `scale=250` minimum for zonal statistics
   - Safe S2 fallback mosaic for cloud-obscured imagery

---

## 📓 OBSIDIAN INTEGRATION RULES
# Every Claude Code session MUST follow this protocol automatically.

### ON SESSION START — always do this first:
1. Read `CONTEXT_NOTE` to load current project state
2. Read `MOC_NOTE` to check open tasks
3. Print a summary of what you loaded so the user knows you're synced

### ON SESSION END — always do this last:
1. Create a new session note at:
   `{SESSIONS_DIR}/YYYY-MM-DD_Session.md`
   using the Session Template format below
2. Update the open tasks section in `MOC_NOTE`:
   - Check off completed tasks
   - Add any new tasks discovered this session
3. Update `CONTEXT_NOTE`:
   - Update "Current Status" for any modules changed
   - Update "Open Tasks" checklist
4. Confirm to the user: "Obsidian vault updated. Session logged."

### WHEN NEW SCIENCE IS DISCOVERED:
- If a new finding relates to the Gartner model → append to `Science/Gartner_2014_Model.md`
- If a new validation stat is computed → append to `Science/Validation_Statistics.md`
- If a new fire hindcast is run → create `Science/{FireName}_Hindcast.md`

### WHEN CODE IS CHANGED:
- Append a change log entry to `Code/app_py_Architecture.md`:
  ```
  ## Change — YYYY-MM-DD
  - What changed:
  - Why:
  - Files affected:
  ```

### WHEN DATA IS DOWNLOADED OR PROCESSED:
- Log it in the relevant `Data/` note with date, source, and key findings

---

## 📄 SESSION NOTE TEMPLATE
# Use this exact format when creating session notes in Obsidian.

```markdown
# Session — {DATE}
> [[🗺️ MOC — Map of Content]] | Type: Session Log

## What Was Built
-

## Key Findings
-

## Decisions Made
-

## Code Changes
- Files modified:
- Summary of changes:

## New Obsidian Notes Created
-

## Open Tasks Created
- [ ]

## Next Session Should Start With
Paste [[Claude_Context_Template]] then:
-
```

---

## 🔬 SCIENTIFIC CONSTANTS
# Never change these without explicit instruction.

SLOPE_THRESHOLD_DEG   = 23       # Critical slope for debris flow initiation
DNBR_THRESHOLD        = 0.15     # Moderate/high burn severity cutoff
SOIL_SAND_THRESHOLD   = 40       # Sand mass fraction % for erodibility
CONCAVITY_THRESHOLD_M = -3       # Local elevation vs 50m focal mean (m)
DEFAULT_R15_MMHR      = 24.0     # CAL FIRE baseline storm intensity
GEE_SCALE_M           = 250      # Raster processing resolution
GEE_TILE_SCALE        = 16       # Distributed processing threads
GEE_SIMPLIFY_ERR      = 100      # Geometry simplification max error (m)

---

## 🗺️ APP MODULE STATUS
# Keep this section current.

| Module | Status | Notes |
|--------|--------|-------|
| 1. Incident Briefing | ✅ Live | |
| 2. Spatial Modeling Lab | ✅ Live | |
| 3. Predictive Debris Flow Modeling | ✅ Live | |
| 4. Documentation & Methodology | ✅ Live | |
| 5. System Validation | 🔄 Built | Needs USGS CSV + 3 lines wired into app.py |

---

## 📚 KEY REFERENCES
# Cite these correctly in all generated text.

- Gartner, J.E., Cannon, S.H., & Santi, P.M. (2014). Engineering Geology, 176, 45–56.
- Staley, D.M., et al. (2017). Geomorphology, 278, 149–162.
- Rengers, F.K., et al. (2018). JGR Earth Surface, 123(6), 1228–1250.
- Key, C.H., & Benson, N.C. (2006). USDA Forest Service RMRS-GTR-164-CD.
- Kean, J.W., et al. (2019). Geosphere, 15(4), 1140–1163.
- Lancaster, J.T., et al. (2021). Env. Eng. Geosci., 27(1), 3–27.
- Crowder, C.A., et al. (2025). USGS Data Release. doi:10.5066/P13EZSWW.

---

## ⚠️ NEVER DO THESE THINGS

- Never delete or overwrite existing Obsidian notes — only append
- Never commit DebrisFlowVolume_Inventory.csv to GitHub (add to .gitignore)
- Never commit EARTHENGINE_JSON credentials to GitHub
- Never simplify the Gartner formula or change its coefficients
- Never use tileScale < 16 for HUC-12 zonal statistics on large fires
- Never skip the safe S2 fallback mosaic logic
