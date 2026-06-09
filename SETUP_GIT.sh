# EOCE Physical Model

**Extreme Orographic Convective Events (EOCE) — 3D Bin-Microphysics Simulation**

[![arXiv](https://img.shields.io/badge/arXiv-2411.08219-b31b1b.svg)](https://arxiv.org/abs/2411.08219)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Scientific Reports](https://img.shields.io/badge/Submitted-Scientific%20Reports-blue.svg)]()

> Haas, R. (2024). *Extreme Orographic Convective Events (EOCE): Physical Modelling of Canyon Supercell Collapse, Amphitheater Erosion, and Implications for Persistent Geological Anomalies.* arXiv:2411.08219

---

## Overview

This repository contains the open-source numerical model supporting the EOCE hypothesis. An EOCE (known as *toroh* in the Tupi-Guarani oral tradition) is a coherent hydraulic ice-piston that forms when a decaying supercell with anomalously narrow drop size distribution (γ shape parameter μ ≈ 20) undergoes explosive secondary ice production via the Hallett-Mossop mechanism, triggered by marine iodine INPs injected by a concurrent Rankine vortex, and collapses coherently into canyon terrain.

### Physical Chain Modelled

```
Stage 1 — Narrow-DSD supercell (μ = 20): iridescent precursor cloud
Stage 2 — Rankine vortex (V_max = 70 m/s): marine INP injection
Stage 3 — Hallett-Mossop SIP cascade (350 splinters/mg, -3°C to -8°C)
Stage 4 — Hydraulic piston formation (f_ice > 0.6, sintering cohesion)
Stage 5 — Coherent collapse (Joukowsky: P = ρ_mix · c_sound · v_impact)
Stage 6 — Selective Shields erosion (zero fine-matrix diagnostic signature)
```

### Key Results (Reference Event: 10 m³ piston, 2 km fall)

| Parameter | Value |
|-----------|-------|
| Impact velocity | ~198 m/s |
| Joukowsky pressure | ~63 MPa |
| Total energy | ~7.4 × 10⁹ J |
| Seismic equivalent | M_L 3.4 |
| Fine sediment removed | 100% (clay, silt, fine sand) |

---

## Installation

```bash
git clone https://github.com/reinaldohaas/eoce-model.git
cd eoce-model
pip install -r requirements.txt
```

**Requirements:** Python ≥ 3.9, NumPy ≥ 1.24, Matplotlib ≥ 3.7, SciPy ≥ 1.10

---

## Usage

```bash
# Run with default parameters (50×50×80 grid, 60 time steps)
python eoce_model.py

# Custom grid resolution
python eoce_model.py --nx 80 --ny 80 --nz 120 --nt 90

# Custom output directory
python eoce_model.py --out my_results/
```

---

## Output Figures

All figures are saved to `./eoce_output/`:

| Figure | Description |
|--------|-------------|
| `fig1_dsd.png` | Narrow gamma DSD (μ=20) vs. conventional + iridescence |
| `fig2_vortex.png` | Rankine vortex velocity field + marine INP injection |
| `fig3_sip_cascade.png` | Hallett-Mossop SIP cascade time evolution (cross-sections) |
| `fig4_piston_3d.png` | Hydraulic piston 3D structure at collapse |
| `fig5_cohesion.png` | Sintering cohesion criterion vs. aerodynamic fragmentation |
| `fig6_impact.png` | Joukowsky impact pressure map at ground level |
| `fig7_shields.png` | Shields selective erosion by grain size class |
| `fig8_energy.png` | Energy budget + seismic equivalence |

---

## Physical Parameters

### Hallett-Mossop Secondary Ice Production
- Active zone: −3°C to −8°C
- Rate: 350 ice splinters per mg rime accreted (Hallett & Mossop 1974)
- Requires: droplets < 13 μm AND droplets > 24 μm simultaneously
- Supplemented by Phillips et al. (2017) collisional breakup at −15°C

### Piston Cohesion (Sintering Criterion)
- Stokes relaxation time τ_p = ρ_p d² / (18 μ_air) ≈ 0.04 s
- Ice sintering time τ_sint = 1–10 s (Szabo & Schneebeli 2007)
- Since τ_p << τ_sint: inter-particle velocities small enough for bond formation
- Sintering tensile strength: ~10⁴ Pa (Kermani et al. 2008)
- Aerodynamic fragmentation pressure at 40 m/s: ~10³ Pa
- Safety margin: ~1 order of magnitude → piston remains coherent

### Joukowsky Impact Pressure
```
P_impact = ρ_mix × c_sound_mix × v_impact
P_impact = 800 kg/m³ × 400 m/s × 198 m/s ≈ 63 MPa
```

### Shields Selective Erosion
```
τ_critical = θ_c × (ρ_s - ρ_w) × g × d₅₀
θ_c = 0.047 (standard Shields parameter)
```
All grains below coarse sand threshold are completely mobilized.
**Result: zero clay/silt in erosion scar** — diagnostic EOCE signature.

---

## Observational Calibration

Calibrated to **Vale do Revólver**, Presidente Getúlio, SC, Brazil:
- Coordinates: 26.89°S, 49.37°W
- Canyon elevation: ~200 m
- Dominant air mass: South Atlantic oceanic
- Observed: two-phase acoustic signature, M_L 2–3 seismicity,
  linear erosion scar with zero clay residue, iridescent precursor clouds

See `data/vale_do_revolver_params.json` for full observational parameters.

---

## Scientific Context

### Four Geological Anomalies Addressed

1. **Amphitheater erosion in resistant bedrock** — sapping hypothesis rejected
   (Lamb et al. 2007, GSA Bulletin): EOCE hydraulic jet provides mechanism
2. **Heavy mineral concentration in canyon lag deposits** — selective Shields
   washing concentrates dense minerals (placer-like mechanism)
3. **Great Unconformity heterogeneity** — topographically controlled, episodic,
   fine-selective erosion incompatible with Snowball Earth glaciation
4. **Cambrian Explosion nutrient pulse** — EOCE erosion of fresh basement rock
   delivers P, Fe, K to adjacent marine environments

### Temporal Modulators

- **Tinsley mechanism**: solar wind → atmospheric Jz → cloud microphysics
- **Forbush Decrease**: CME sweeps GCR, favoring EOCE formation
- **Adams Event (Laschamp ~42 ka)**: geomagnetic collapse → two-phase drought/EOCE
- **Anthropogenic suppression**: TEL lead aerosols suppressed EOCE in 20th century

---

## References

- Hallett, J. & Mossop, S.C. (1974). Production of secondary ice particles during the riming process. *Nature*, 249, 26–28.
- Phillips, V.T.J. et al. (2017). Ice multiplication by breakup in ice-ice collisions. *J. Atmos. Sci.*, 74(5), 1705–1719.
- Lamb, M.P. et al. (2007). Formation of amphitheater-headed valleys by waterfall erosion. *GSA Bulletin*, 119(7–8), 805–822.
- Szabo, D. & Schneebeli, M. (2007). Subsecond sintering of ice. *Appl. Phys. Lett.*, 90(15), 151916.
- Kermani, M. et al. (2008). Mechanical properties of ice under high strain rates. *Cold Reg. Sci. Technol.*, 54(3), 183–191.
- Tinsley, B.A. (2000). Influence of solar wind on the global electric circuit. *Space Sci. Rev.*, 94, 231–258.
- Cooper, A. et al. (2021). A global environmental crisis 42,000 years ago. *Science*, 371(6531), 811–818.
- Keller, C.B. et al. (2019). Neoproterozoic glacial origin of the Great Unconformity. *PNAS*, 116(4), 1136–1145.

---

## Citation

```bibtex
@article{haas2024eoce,
  author  = {Haas, Reinaldo},
  title   = {Extreme Orographic Convective Events (EOCE): Physical Modelling
             of Canyon Supercell Collapse, Amphitheater Erosion, and
             Implications for Persistent Geological Anomalies},
  year    = {2024},
  eprint  = {2411.08219},
  archivePrefix = {arXiv},
  url     = {https://arxiv.org/abs/2411.08219}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

**Author:** Reinaldo Haas  
**Affiliation:** Department of Physics, UFSC, Florianópolis, Brazil  
**Contact:** reinaldo.haas@ufsc.br
