# Doppler Visibility Map

Interactive radar Doppler visibility map — visualise blind speeds and detection coverage across staggered PRFs.

---

## Files

| File | Description |
|------|-------------|
| `doppler_visibility.py` | Core computation engine (Python/NumPy) |
| `plot_static.py` | Static Matplotlib plotter — closest to MATLAB output |
| `index.html` | **Interactive web app — works on phone, laptop, anywhere** |
| `requirements.txt` | Python dependencies |

---

## Quick Start

### Web App (recommended — works on phone & laptop)
Just open `index.html` in any browser. No server, no install.

Controls:
- **Band preset** or **manual frequency** entry
- **6 PRF sliders** — drag to adjust each PRF in real time
- **Blind zone width** slider — controls how wide each blind speed zone is
- **Tabs** — Visibility map / Frequency domain / Coverage bar chart
- **Presets** — 3-PRF stagger, 6-PRF stagger, Airborne, Maritime, Ground Map, Low PRF
- **Export CSV** — downloads the full visibility matrix

### Python Static Plot (MATLAB-equivalent)
```bash
pip install -r requirements.txt
python plot_static.py
```
Saves `doppler_visibility_map.png` in the current directory.

### Python Computation Only
```python
from doppler_visibility import DopplerVisibility

dv = DopplerVisibility(
    f_radar=10e9,                               # 10 GHz X-band
    prfs=[1000, 1250, 1500, 1750, 2000, 2500],  # Hz
    v_range=(-600, 600),                        # m/s
    blind_fraction=0.05,                        # 5% blind zone width
)

result = dv.compute()
print(dv.summary(result))

# result.velocities          — velocity axis
# result.visibility_per_prf  — (n_prfs, N) binary matrix
# result.combined_visibility — (N,) AND of all PRFs
# result.combined_coverage   — fraction visible by all PRFs
```

---

## Physics

**Blind speeds** occur where the target's Doppler shift aliases to DC (zero velocity):

```
v_blind,n = n × λ × PRF / 2     (n = ±1, ±2, ...)
```

**Unambiguous velocity interval:**
```
v_ua = λ × PRF / 2
```

**Stagger ratio:** choosing PRF ratios as simple fractions (e.g. 4:5:6) ensures blind zones
of each PRF do not coincide, maximising combined coverage.

---

## Recommended PRF Stagger Ratios

| Ratio | PRFs (example, 1000 Hz base) | Combined first blind speed |
|-------|------------------------------|---------------------------|
| 4:5   | 1000, 1250                   | ~5× single PRF            |
| 4:5:6 | 1000, 1250, 1500             | ~10× single PRF           |
| 5:6:7 | 1000, 1200, 1400             | ~21× single PRF           |
| Prime | 1000, 1100, 1300, 1700       | very large                |

---

## Requirements

- **Web app:** any modern browser (Chrome, Firefox, Safari, Edge)
- **Python scripts:** Python 3.8+, numpy, matplotlib
