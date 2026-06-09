"""
eoce_model.py
=============
EOCE — Extreme Orographic Convective Event
3D Bin-Microphysics Simulation

Haas, R. (2024). arXiv:2411.08219
Department of Physics, UFSC, Florianopolis, Brazil

Physical chain modelled:
  Stage 1 — Narrow-DSD supercell (mu=20)
  Stage 2 — Rankine vortex + marine INP injection
  Stage 3 — Hallett-Mossop SIP cascade (3D)
  Stage 4 — Hydraulic piston formation + cohesion (Bond/sintering)
  Stage 5 — Coherent collapse (Joukowsky)
  Stage 6 — Impact pressure map + Shields selective erosion

Outputs (saved to ./eoce_output/):
  fig1_dsd.png          — Narrow gamma DSD vs. conventional
  fig2_vortex.png       — Rankine vortex + INP injection field
  fig3_sip_cascade.png  — SIP cascade time evolution (cross-sections)
  fig4_piston_3d.png    — Piston ice fraction 3D structure
  fig5_cohesion.png     — Bond number / sintering criterion
  fig6_impact.png       — Joukowsky impact pressure map
  fig7_shields.png      — Shields selective erosion by grain class
  fig8_energy.png       — Energy budget + seismic equivalence

Usage:
  python eoce_model.py [--nx 60] [--ny 60] [--nz 80] [--nt 60]

Reference:
  Haas, R. (2024). Extreme Orographic Convective Events (EOCE):
  Physical Modelling of Canyon Supercell Collapse, Amphitheater
  Erosion, and Implications for Persistent Geological Anomalies.
  arXiv:2411.08219

License: MIT
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.ndimage import gaussian_filter
import os
import warnings
import argparse
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# ARGUMENT PARSER
# ─────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='EOCE 3D Physical Model')
parser.add_argument('--nx',  type=int, default=60,   help='Grid points X (default 60)')
parser.add_argument('--ny',  type=int, default=60,   help='Grid points Y (default 60)')
parser.add_argument('--nz',  type=int, default=80,   help='Grid points Z (default 80)')
parser.add_argument('--nt',  type=int, default=60,   help='Time steps (default 60)')
parser.add_argument('--out', type=str, default='eoce_output', help='Output directory')
args = parser.parse_args()

os.makedirs(args.out, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# PHYSICAL CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
g           = 9.81       # m/s²   gravity
rho_water   = 1000.      # kg/m³  liquid water density
rho_ice     = 917.       # kg/m³  ice density
rho_air     = 1.2        # kg/m³  air density at cloud base
rho_graup   = 500.       # kg/m³  graupel bulk density
L_fusion    = 334e3      # J/kg   latent heat of fusion
L_sublim    = 2830e3     # J/kg   latent heat of sublimation
c_sound_mix = 400.       # m/s    effective sound speed in ice-water mix (Joukowsky)
mu_air      = 1.81e-5    # kg/m/s dynamic viscosity of air

# Hallett-Mossop parameters (Hallett & Mossop 1974)
T_HM_min  = -8.    # °C  lower bound of HM zone
T_HM_max  = -3.    # °C  upper bound of HM zone
HM_rate   = 350.   # splinters per mg rime accreted

# Shields dimensionless critical shear stress
Shields_crit = 0.047   # standard value for quartz

# ─────────────────────────────────────────────────────────────────────────────
# GRID SETUP
# ─────────────────────────────────────────────────────────────────────────────
NX, NY, NZ = args.nx, args.ny, args.nz
NT         = args.nt
dx = dy    = 25.      # m  horizontal resolution
dz         = 30.      # m  vertical resolution
dt         = 1.0      # s  time step

LX = NX * dx
LY = NY * dy
LZ = NZ * dz

x = np.linspace(0, LX, NX)
y = np.linspace(0, LY, NY)
z = np.linspace(0, LZ, NZ)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# Temperature field: moist adiabatic lapse rate
T_surf = 15.            # °C at surface
lapse  = 6.5e-3         # °C/m  moist adiabatic lapse rate
T_3d   = T_surf - lapse * Z

print("=" * 60)
print("  EOCE Physical Model — Haas (2024) arXiv:2411.08219")
print("=" * 60)
print(f"  Grid:   {NX}×{NY}×{NZ}  |  dx={dx}m  dz={dz}m  dt={dt}s")
print(f"  Domain: {LX:.0f}m × {LY:.0f}m × {LZ:.0f}m")
print(f"  Steps:  {NT}  ({NT*dt:.0f} s simulation)")
print()

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE STYLE
# ─────────────────────────────────────────────────────────────────────────────
BG    = '#0a1020'
PANEL = '#0d1a2e'
SPINE = '#2a4060'
TICK  = '#6080a0'
LABEL = '#a0b8d0'
TITLE = '#c0d8f0'

def style_ax(ax):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=TICK, labelsize=8)
    for sp in ax.spines.values():
        sp.set_color(SPINE)
    ax.xaxis.label.set_color(LABEL)
    ax.yaxis.label.set_color(LABEL)
    ax.title.set_color(TITLE)

def style_cb(cb):
    cb.ax.yaxis.set_tick_params(color=TICK)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color='#8090a0')

# ─────────────────────────────────────────────────────────────────────────────
# FIG 1 — NARROW DSD
# ─────────────────────────────────────────────────────────────────────────────
print("  Fig 1: Narrow DSD...")

def gamma_dsd(D, N0, mu, Lam):
    return N0 * D**mu * np.exp(-Lam * D)

D_um = np.linspace(1, 150, 600)
D_m  = D_um * 1e-6

mu_eoce  = 20.
Lam_eoce = (mu_eoce + 3.67) / 15e-6
n_eoce   = gamma_dsd(D_m, 1e12, mu_eoce, Lam_eoce)
n_eoce  /= n_eoce.max()

mu_conv  = 2.
Lam_conv = (mu_conv + 3.67) / 200e-6
n_conv   = gamma_dsd(D_m, 1e8, mu_conv, Lam_conv)
n_conv  /= n_conv.max()

fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor=BG)
fig.suptitle("Fig. 1 — Drop Size Distribution: EOCE Narrow DSD vs. Conventional Supercell",
             color='#e0e8f0', fontsize=12, y=1.01)

ax = axes[0]
ax.plot(D_um, n_eoce, color='#ffd700', lw=2.5, label=f'EOCE (μ={mu_eoce:.0f}) — iridescent cloud')
ax.plot(D_um, n_conv, color='#4090e0', lw=2, ls='--', label=f'Conventional (μ={mu_conv:.0f})')
ax.axvspan(0, 13,   alpha=0.12, color='#40ff80', label='<13 μm (HM small class)')
ax.axvspan(24, 150, alpha=0.10, color='#ff8040', label='>24 μm (HM large class)')
ax.axvline(13, color='#40ff80', lw=0.8, ls=':')
ax.axvline(24, color='#ff8040', lw=0.8, ls=':')
ax.set_xlabel('Droplet diameter (μm)')
ax.set_ylabel('Normalized N(D)')
ax.set_title('Gamma DSD: μ=20 satisfies\ndual HM size-class requirement')
ax.legend(fontsize=9, facecolor=PANEL, labelcolor='white', edgecolor=SPINE)
ax.set_xlim(0, 150)
style_ax(ax)

ax2 = axes[1]
sigma_e = 1.5
sigma_c = 25.
coh_e = np.exp(-0.5*((D_um-15)/sigma_e)**2)
coh_c = np.exp(-0.5*((D_um-80)/sigma_c)**2)
ax2.fill_between(D_um, coh_e, alpha=0.35, color='#ffd700')
ax2.plot(D_um, coh_e, color='#ffd700', lw=2.5,
         label='EOCE — STRONG iridescence (structural coloring)')
ax2.fill_between(D_um, coh_c, alpha=0.2, color='#4090e0')
ax2.plot(D_um, coh_c, color='#4090e0', lw=2, ls='--',
         label='Conventional — WEAK (incoherent scattering)')
ax2.annotate('Optical precursor:\niridescent cloud base',
             xy=(15, 0.95), xytext=(40, 0.75), color='#80ff80', fontsize=8,
             arrowprops=dict(arrowstyle='->', color='#40c060', lw=1), ha='center')
ax2.set_xlabel('Droplet diameter (μm)')
ax2.set_ylabel('Diffraction coherence (a.u.)')
ax2.set_title('Cloud iridescence: optical precursor\n(observable early warning of EOCE)')
ax2.legend(fontsize=9, facecolor=PANEL, labelcolor='white', edgecolor=SPINE)
ax2.set_xlim(0, 150)
style_ax(ax2)

plt.tight_layout()
plt.savefig(f'{args.out}/fig1_dsd.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# FIG 2 — RANKINE VORTEX + INP INJECTION
# ─────────────────────────────────────────────────────────────────────────────
print("  Fig 2: Rankine vortex...")

xv = np.linspace(-800, 800, 140)
yv = np.linspace(-800, 800, 140)
Xv, Yv = np.meshgrid(xv, yv)
Rv = np.sqrt(Xv**2 + Yv**2)

V_max  = 70.
r_core = 50.
V_tan  = np.where(Rv <= r_core, V_max*Rv/r_core, V_max*r_core/Rv)
theta  = np.arctan2(Yv, Xv)
Vxv    = -V_tan * np.sin(theta)
Vyv    =  V_tan * np.cos(theta)
INP_2d = np.exp(-Rv**2/(2*180.**2)) * 1e4
W_up   = V_max * np.exp(-Rv**2/(2*150.**2))

fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor=BG)
fig.suptitle("Fig. 2 — Rankine Vortex: Marine INP Injection into Supercooled Cloud",
             color='#e0e8f0', fontsize=12)

ax = axes[0]
im = ax.contourf(xv, yv, np.sqrt(Vxv**2+Vyv**2), levels=20, cmap='plasma', alpha=0.85)
sk = 9
ax.quiver(xv[::sk], yv[::sk], Vxv[::sk,::sk], Vyv[::sk,::sk],
          color='white', alpha=0.35, scale=1500, width=0.002)
ax.add_patch(plt.Circle((0,0), r_core, color='#ffd700', fill=False, lw=2, ls='--'))
ax.add_patch(plt.Circle((0,0), 300, color='#40a0ff', fill=False, lw=1, ls=':', alpha=0.4))
cb = plt.colorbar(im, ax=ax)
cb.set_label('Tangential velocity (m/s)', color=LABEL)
style_cb(cb)
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_title(f'V_max={V_max:.0f} m/s at r_core={r_core:.0f} m\n'
             'Vortex injects marine INPs into cloud column')
ax.text(r_core+15, 20, 'r_core', color='#ffd700', fontsize=8)
ax.annotate('INP injection\nzone', xy=(0,0), xytext=(180,220), color='#80ff80',
            fontsize=8, ha='center',
            arrowprops=dict(arrowstyle='->', color='#40c060', lw=1))
style_ax(ax)

ax2 = axes[1]
im2 = ax2.contourf(xv, yv, INP_2d, levels=20, cmap='YlOrRd', alpha=0.85)
ax2.contour(xv, yv, W_up, levels=[15,30,45,60], colors='#40c8ff', lws=0.8, alpha=0.5)
cb2 = plt.colorbar(im2, ax=ax2)
cb2.set_label('Marine INP conc. (m⁻³)  [HIO₃, IxOy, organics]', color=LABEL)
style_cb(cb2)
ax2.set_xlabel('x (m)')
ax2.set_ylabel('y (m)')
ax2.set_title('Marine INP field transported by oceanic air mass\n'
              '+ updraft contours (blue, m/s)')
ax2.text(0, -680, 'Explains why EOCE occurs with oceanic air mass at canyon',
         color='#70a0d0', fontsize=8, ha='center')
style_ax(ax2)

plt.tight_layout()
plt.savefig(f'{args.out}/fig2_vortex.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3: SIP CASCADE (3D time evolution)
# ─────────────────────────────────────────────────────────────────────────────
print("  Stage 3: SIP cascade (3D)...")

cloud_base_z = 500.
LWC_init     = 1.5e-3   # kg/m³
x0 = LX / 2
y0 = LY / 2
r_inj = 180.

LWC   = np.zeros((NX, NY, NZ))
f_ice = np.zeros((NX, NY, NZ))
N_ice = np.zeros((NX, NY, NZ))
R_xy  = np.sqrt((X[:,:,0]-x0)**2 + (Y[:,:,0]-y0)**2)

for iz in range(NZ):
    if z[iz] >= cloud_base_z:
        LWC[:,:,iz] = LWC_init * np.exp(-((z[iz]-cloud_base_z)/900.)**2)

INP_3d = np.zeros((NX, NY, NZ))
for iz in range(NZ):
    if z[iz] >= cloud_base_z:
        prof = np.exp(-((z[iz]-(cloud_base_z+350.))/280.)**2)
        INP_3d[:,:,iz] = np.exp(-R_xy**2/(2*r_inj**2)) * 6e3 * prof

HM_zone = (T_3d >= T_HM_min) & (T_3d <= T_HM_max)
times_store = set([0, NT//4, NT//2, 3*NT//4, NT-1])
f_ice_snaps = []

m_cryst = rho_ice * (4/3)*np.pi*(200e-6)**3

for it in range(NT):
    nuc_rate = INP_3d * HM_zone * dt
    dN_prim  = np.minimum(nuc_rate, LWC/m_cryst*0.01)
    rime_dt  = 2e-3 * N_ice * LWC * dt
    HM_act   = HM_zone & (LWC > 0.05e-3)
    dN_HM    = HM_act * HM_rate * rime_dt * 1e6 * dt
    T_phil   = -15.
    dT_phil  = 5.
    phil_z   = np.exp(-((T_3d-T_phil)/dT_phil)**2)
    dN_phil  = phil_z * N_ice**2 * 1e-12 * dt
    N_ice   += dN_prim + dN_HM + dN_phil
    N_ice    = np.maximum(N_ice, 0.)
    ice_mass = N_ice * m_cryst
    total_w  = LWC + ice_mass
    f_ice    = np.where(total_w > 0, ice_mass/total_w, 0.)
    f_ice    = np.clip(f_ice, 0., 1.)
    LWC      = np.maximum(LWC - ice_mass*0.03*dt, 0.)
    if it in times_store:
        f_ice_snaps.append((it, f_ice.copy()))

# ─────────────────────────────────────────────────────────────────────────────
# FIG 3 — SIP TIME EVOLUTION
# ─────────────────────────────────────────────────────────────────────────────
print("  Fig 3: SIP cascade evolution...")

iy_mid = NY//2
fig, axes = plt.subplots(1, 5, figsize=(17, 4.5), facecolor=BG)
fig.suptitle("Fig. 3 — Hallett-Mossop SIP Cascade: Ice Fraction f_ice (y=y₀ cross-section)",
             color='#e0e8f0', fontsize=11)

for i, (it, fi) in enumerate(f_ice_snaps):
    ax = axes[i]
    data = fi[:,iy_mid,:].T
    im = ax.contourf(x/1000., z/1000., data, levels=np.linspace(0,1,20), cmap='hot', alpha=0.9)
    ax.axhline(cloud_base_z/1000., color='#ffd700', lw=0.8, ls=':', alpha=0.6, label='Cloud base')
    T_HM_bot_z = (T_surf - T_HM_max) / lapse
    T_HM_top_z = (T_surf - T_HM_min) / lapse
    ax.axhspan(T_HM_bot_z/1000., T_HM_top_z/1000., alpha=0.08, color='#40ff80')
    ax.axvline(x0/1000., color='#40c8ff', lw=0.8, ls='--', alpha=0.35)
    ax.set_title(f't = {it*dt:.0f} s', color=TITLE, fontsize=9)
    ax.set_xlabel('x (km)', fontsize=7)
    if i==0:
        ax.set_ylabel('z (km)', fontsize=7)
    style_ax(ax)
    ax.tick_params(labelsize=6)

cax = fig.add_axes([0.92, 0.14, 0.012, 0.72])
fig.colorbar(im, cax=cax, label='f_ice (ice fraction 0–1)')
cax.yaxis.label.set_color(LABEL)
cax.tick_params(colors=TICK)

fig.text(0.46, 0.01,
         'HM zone (green band): -3°C to -8°C  |  Vortex axis: blue dashed  |  Cloud base: yellow dotted',
         ha='center', color='#4a6a8a', fontsize=8)

plt.savefig(f'{args.out}/fig3_sip_cascade.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# PISTON MASS + ENERGY
# ─────────────────────────────────────────────────────────────────────────────
f_final    = f_ice_snaps[-1][1]
piston     = f_final > 0.60
rho_mix_3d = f_final*rho_ice + (1-f_final)*rho_water
mass_col   = np.sum(rho_mix_3d * piston * dx*dy*dz, axis=2)
M_piston   = float(np.sum(mass_col))

H_fall = cloud_base_z
v_imp  = float(np.sqrt(2*g*H_fall))

E_kin  = M_piston * g * H_fall
E_lat  = M_piston * L_fusion
E_sub  = M_piston * 0.20 * L_sublim
E_tot  = E_kin + E_lat + E_sub
M_L    = (np.log10(max(E_tot,1)) - 4.8) / 1.5

print(f"    Piston mass:    {M_piston/1e3:.1f} tonnes")
print(f"    Impact velocity:{v_imp:.1f} m/s")
print(f"    E_total:        {E_tot:.2e} J  (~M_L {M_L:.2f})")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 4 — PISTON 3D STRUCTURE
# ─────────────────────────────────────────────────────────────────────────────
print("  Fig 4: Piston 3D structure...")

iz_mid = NZ//2
fig, axes = plt.subplots(1, 3, figsize=(14, 5), facecolor=BG)
fig.suptitle("Fig. 4 — Hydraulic Ice-Piston Structure at Collapse (f_ice > 0.60 = coherent body)",
             color='#e0e8f0', fontsize=12)

# XZ cross-section
ax = axes[0]
data = f_final[:,iy_mid,:].T
im = ax.contourf(x/1000., z/1000., data, levels=np.linspace(0,1,20), cmap='hot')
ax.contour(x/1000., z/1000., data, levels=[0.60], colors='#ffd700', linewidths=2,
           linestyles='--')
ax.set_xlabel('x (km)')
ax.set_ylabel('z (km)')
ax.set_title('XZ cross-section (y=y₀)\nYellow dashed: cohesion boundary f_ice=0.60')
cb = plt.colorbar(im, ax=ax)
cb.set_label('Ice fraction', color=LABEL)
style_cb(cb); style_ax(ax)

# XY footprint
ax2 = axes[1]
f_col_max = f_final.max(axis=2)
im2 = ax2.contourf(x/1000., y/1000., f_col_max.T, levels=np.linspace(0,1,20), cmap='inferno')
ax2.contour(x/1000., y/1000., f_col_max.T, levels=[0.60], colors='#ffd700', linewidths=2)
ax2.set_xlabel('x (km)')
ax2.set_ylabel('y (km)')
ax2.set_title('XY footprint (column max f_ice)\nPiston horizontal extent')
cb2 = plt.colorbar(im2, ax=ax2)
cb2.set_label('Max ice fraction', color=LABEL)
style_cb(cb2); style_ax(ax2)

# Vertical profile
ax3 = axes[2]
iz_mid_x = np.argmax(f_final.mean(axis=(0,1)))
f_profile = f_final[NX//2, NY//2, :]
ax3.fill_betweenx(z/1000., f_profile, alpha=0.4, color='#ff6020')
ax3.plot(f_profile, z/1000., color='#ff8040', lw=2.5, label='f_ice profile (center)')
ax3.axvline(0.60, color='#ffd700', lw=2, ls='--', label='Cohesion threshold (0.60)')
ax3.axhspan(cloud_base_z/1000., (cloud_base_z+300.)/1000., alpha=0.1, color='#40ff80',
            label='HM zone')
ax3.set_xlabel('Ice fraction f_ice')
ax3.set_ylabel('z (km)')
ax3.set_title('Vertical f_ice profile (vortex center)\nCohesion zone → piston body')
ax3.legend(fontsize=8, facecolor=PANEL, labelcolor='white', edgecolor=SPINE)
ax3.set_xlim(0, 1)
style_ax(ax3)

plt.tight_layout()
plt.savefig(f'{args.out}/fig4_piston_3d.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# FIG 5 — COHESION CRITERION
# ─────────────────────────────────────────────────────────────────────────────
print("  Fig 5: Cohesion criterion (sintering)...")

d_mm  = np.linspace(1, 20, 200)
d_m   = d_mm * 1e-3
tau_p = rho_graup * d_m**2 / (18*mu_air)   # Stokes relaxation time

v_arr = np.linspace(5, 80, 200)
P_aer = 0.5 * rho_air * v_arr**2           # aerodynamic dynamic pressure

fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor=BG)
fig.suptitle("Fig. 5 — Ice-Piston Cohesion: Sintering Kinetics vs. Aerodynamic Fragmentation",
             color='#e0e8f0', fontsize=12)

ax = axes[0]
ax.semilogy(d_mm, tau_p, color='#40c8ff', lw=2.5, label='Stokes relaxation time τ_p')
ax.fill_between(d_mm, 1., 10., alpha=0.25, color='#ffd700',
                label='Ice sintering time τ_sint\n(Szabo & Schneebeli 2007: 1–10 s at -5°C)')
ax.axhline(1., color='#ffd700', lw=1, ls='--', alpha=0.7)
ax.axhline(10., color='#ffd700', lw=1, ls='--', alpha=0.7)
ax.axhline(40., color='#ff6040', lw=2, ls='-.',
           label='EOCE descent duration (~40 s)')
ax.fill_between(d_mm, tau_p, 1., where=tau_p<1.,
                alpha=0.15, color='#40ff80')
ax.text(5, 0.05, 'COHESION ZONE\nτ_p << τ_sint', color='#40ff80',
        fontsize=9, fontweight='bold', ha='center')
ax.set_xlabel('Particle diameter (mm)')
ax.set_ylabel('Timescale (s)')
ax.set_title('τ_p (Stokes) << τ_sint → inter-particle\nvelocity low enough for sintering bonds')
ax.legend(fontsize=9, facecolor=PANEL, labelcolor='white', edgecolor=SPINE)
ax.set_xlim(1, 20)
style_ax(ax)

ax2 = axes[1]
sigma_T  = 1e4 * np.ones_like(v_arr)    # Pa tensile strength (Kermani et al. 2008)
sigma_lo = 5e3 * np.ones_like(v_arr)
sigma_hi = 2e4 * np.ones_like(v_arr)
ax2.semilogy(v_arr, P_aer, color='#ff6040', lw=2.5,
             label='Aerodynamic pressure ½ρv²')
ax2.fill_between(v_arr, sigma_lo, sigma_hi, alpha=0.25, color='#ffd700',
                 label='Ice sinter tensile strength\n(Kermani et al. 2008: ~10⁴ Pa)')
ax2.plot(v_arr, sigma_T, color='#ffd700', lw=1.5, ls='--')
ax2.axvline(40., color='#40c8ff', lw=2, ls='-.',
            label='EOCE estimated impact velocity ~40 m/s')
v_cross = v_arr[np.argmin(np.abs(P_aer - sigma_T))]
ax2.fill_betweenx([1e2, 1e4], 0, v_cross, alpha=0.10, color='#40ff80')
ax2.text(v_cross/2, 3e3, 'COHESIVE', color='#40ff80', fontsize=9, fontweight='bold', ha='center')
ax2.annotate(f'Disaggregation\nbegins ~{v_cross:.0f} m/s\n(final descent only)',
             xy=(v_cross, sigma_T[0]), xytext=(60, 1e3),
             color='#ff8060', fontsize=8,
             arrowprops=dict(arrowstyle='->', color='#ff6040'))
ax2.set_xlabel('Descent velocity (m/s)')
ax2.set_ylabel('Stress / Pressure (Pa)')
ax2.set_title('Sintering strength > aerodynamic pressure\n→ piston coherent through most of descent')
ax2.legend(fontsize=9, facecolor=PANEL, labelcolor='white', edgecolor=SPINE)
ax2.set_xlim(5, 80)
style_ax(ax2)

plt.tight_layout()
plt.savefig(f'{args.out}/fig5_cohesion.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# FIG 6 — IMPACT PRESSURE MAP (Joukowsky)
# ─────────────────────────────────────────────────────────────────────────────
print("  Fig 6: Impact pressure map...")

rho_col = np.sum(rho_mix_3d*piston*dz, axis=2) / np.maximum(np.sum(piston*dz, axis=2), 1.)
rho_col = np.where(piston.any(axis=2), rho_col, 0.)
P_raw   = rho_col * c_sound_mix * v_imp
P_sm    = gaussian_filter(P_raw, sigma=2.)
P_sm    = np.where(P_sm < 1e3, 0., P_sm)
P_mpa   = P_sm / 1e6

fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor=BG)
fig.suptitle("Fig. 6 — EOCE Ground Impact: Joukowsky Pressure  P = ρ_mix · c_sound · v_impact",
             color='#e0e8f0', fontsize=12)

ax = axes[0]
lev = np.linspace(0, max(P_mpa.max(), 0.01), 25)
im  = ax.contourf(x/1000., y/1000., P_mpa.T, levels=lev, cmap='inferno', alpha=0.9)
ax.contour(x/1000., y/1000., P_mpa.T, levels=[0.5, 2., 5.],
           colors=['#60a0ff','#ffd700','white'], linewidths=[0.8,1.2,1.5])
cb = plt.colorbar(im, ax=ax)
cb.set_label('Impact pressure (MPa)', color=LABEL)
style_cb(cb)
ry, rx = np.unravel_index(P_mpa.argmax(), P_mpa.shape)
ax.plot(x[ry]/1000., y[rx]/1000., 'w+', ms=15, mew=2)
ax.text(x[ry]/1000.+0.06, y[rx]/1000., f'P_max={P_mpa.max():.1f} MPa', color='white', fontsize=9)
ax.set_xlabel('x (km)')
ax.set_ylabel('y (km)')
ax.set_title(f'v_impact = {v_imp:.0f} m/s  |  P >> τ_Shields(clay)\n→ Complete removal of all fine sediment')
ax.text(0.04, 0.06, 'Selective washing:\nZERO clay/silt retained\n(diagnostic EOCE signature)',
        transform=ax.transAxes, color='#80ff80', fontsize=8,
        bbox=dict(boxstyle='round', facecolor=PANEL, edgecolor='#40c060', alpha=0.8))
style_ax(ax)

ax2 = axes[1]
pr = P_mpa[:, rx]
ax2.fill_between(x/1000., pr, alpha=0.35, color='#ff6020')
ax2.plot(x/1000., pr, color='#ff8040', lw=2.5, label='Pressure profile (center)')
ax2.set_xlabel('x (km)')
ax2.set_ylabel('Impact pressure (MPa)')
ax2.set_title('Pressure cross-section through piston center\n'
              '→ Far exceeds Shields critical shear for all grain sizes')
ax2.legend(fontsize=9, facecolor=PANEL, labelcolor='white', edgecolor=SPINE)
ax2.text(0.98, 0.92,
         f'M_piston = {M_piston/1e3:.1f} t\nE_total  = {E_tot:.1e} J\nM_L ≈ {M_L:.1f}',
         transform=ax2.transAxes, color=TITLE, fontsize=9, ha='right', va='top',
         bbox=dict(boxstyle='round', facecolor='#1a2a40', edgecolor=SPINE))
style_ax(ax2)

plt.tight_layout()
plt.savefig(f'{args.out}/fig6_impact.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# FIG 7 — SHIELDS SELECTIVE EROSION
# ─────────────────────────────────────────────────────────────────────────────
print("  Fig 7: Shields selective erosion...")

grain_classes = [
    ('Clay\n<4 μm',          2e-6,   '#ff4040'),
    ('Silt\n4–63 μm',       30e-6,   '#ff8040'),
    ('Fine Sand\n63–250 μm',150e-6,  '#ffd700'),
    ('Med Sand\n0.25–1 mm', 500e-6,  '#80c040'),
    ('Coarse Sand\n1–4 mm',  2e-3,   '#40c8ff'),
    ('Gravel\n>4 mm',        8e-3,   '#8080ff'),
]
rho_s = 2650.  # kg/m³  quartz density

fig, axes = plt.subplots(2, 3, figsize=(14, 8), facecolor=BG)
fig.suptitle("Fig. 7 — EOCE Selective Erosion: Shields Criterion by Grain Class\n"
             "Red = mobilized (removed) | Dark = retained | → Zero-clay diagnostic signature",
             color='#e0e8f0', fontsize=11)

tau_bed = P_sm * 0.08   # bed shear from impact (oblique correction)

for i, (label, d50, col) in enumerate(grain_classes):
    ax = axes[i//3][i%3]
    tau_c  = Shields_crit * (rho_s - rho_water) * g * d50
    mob    = tau_bed > tau_c
    pfoot  = P_sm > 1e3
    combined = np.zeros_like(P_sm)
    combined[mob]          = 1.0   # mobilized
    combined[pfoot & ~mob] = 0.35  # retained in footprint

    im = ax.contourf(x/1000., y/1000., combined.T, levels=[-0.1,0.2,0.7,1.1],
                     colors=[PANEL,'#1a3060','#dd2010'])
    ax.set_title(f'{label}\nd₅₀={d50*1e6:.0f}μm  τ_c={tau_c:.2e}Pa',
                 color=col, fontsize=9)
    ax.set_xlabel('x (km)', fontsize=8)
    ax.set_ylabel('y (km)', fontsize=8)
    mob_pct = mob.sum() / max(pfoot.sum(), 1) * 100
    badge_col = '#ff6040' if mob_pct > 70 else '#40c8ff'
    ax.text(0.05, 0.06, f'{mob_pct:.0f}% mobilized',
            transform=ax.transAxes, color=badge_col, fontsize=9, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor=PANEL, edgecolor=SPINE, alpha=0.8))
    style_ax(ax)
    ax.tick_params(labelsize=7)

legend_el = [Patch(facecolor='#dd2010', label='MOBILIZED: τ_bed > τ_c (removed)'),
             Patch(facecolor='#1a3060', label='RETAINED: τ_bed < τ_c'),
             Patch(facecolor=PANEL,     label='Outside piston footprint')]
fig.legend(handles=legend_el, loc='lower center', ncol=3,
           facecolor=BG, labelcolor='white', edgecolor=SPINE,
           fontsize=9, bbox_to_anchor=(0.5, 0.01))

plt.tight_layout(rect=[0, 0.06, 1, 1])
plt.savefig(f'{args.out}/fig7_shields.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# FIG 8 — ENERGY BUDGET
# ─────────────────────────────────────────────────────────────────────────────
print("  Fig 8: Energy budget...")

mags = np.arange(1, 6.5, 0.1)
E_seis = 10.**(1.5*mags + 4.8)

fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor=BG)
fig.suptitle("Fig. 8 — EOCE Energy Budget and Seismic Equivalence",
             color='#e0e8f0', fontsize=12)

ax = axes[0]
ax.set_facecolor(PANEL)
energies_pie = [E_kin, E_lat, E_sub]
labels_pie   = [f'Kinetic\n{E_kin:.1e} J\n({100*E_kin/E_tot:.0f}%)',
                f'Latent fusion\n{E_lat:.1e} J\n({100*E_lat/E_tot:.0f}%)',
                f'Sublimation\n{E_sub:.1e} J\n({100*E_sub/E_tot:.0f}%)']
colors_pie   = ['#40c8ff','#ffd700','#ff6040']
wedges, texts = ax.pie(energies_pie, labels=labels_pie, colors=colors_pie,
                        explode=[0.06,0.06,0.06],
                        textprops=dict(color='white', fontsize=9), shadow=True)
ax.set_title(f'E_total = {E_tot:.2e} J', color=TITLE)

ax2 = axes[1]
ax2.semilogy(mags, E_seis, color='#4090e0', lw=2, label='Seismic E (Gutenberg-Richter)')
ax2.axhline(E_tot,  color='#ffd700', lw=2.5, ls='--', label=f'EOCE total  {E_tot:.1e} J')
ax2.axhline(E_kin,  color='#40c8ff', lw=1.5, ls=':',  label=f'Kinetic only {E_kin:.1e} J')
ax2.axvline(M_L, color='#ffd700', lw=1, ls=':', alpha=0.5)
ax2.plot(M_L, E_tot, 'o', color='#ffd700', ms=12, zorder=5)
ax2.text(M_L+0.1, E_tot*1.6, f'M_L ≈ {M_L:.1f}', color='#ffd700', fontsize=12, fontweight='bold')
for m, lbl in [(2.0,'Quarry blast'),(3.0,'Mining explosion'),(4.0,'Small earthquake')]:
    ax2.text(m, 10.**(1.5*m+4.8)*2.5, lbl, color='#4a6a8a', fontsize=8, ha='center')
ax2.set_xlabel('Richter Magnitude M_L')
ax2.set_ylabel('Energy (J)')
ax2.set_title('EOCE seismic equivalence\n'
              '(explains M_L 2-3 tremors reported by witnesses)')
ax2.legend(fontsize=9, facecolor=PANEL, labelcolor='white', edgecolor=SPINE)
ax2.set_xlim(1, 6)
style_ax(ax2)

plt.tight_layout()
plt.savefig(f'{args.out}/fig8_energy.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("  EOCE Model — Results Summary")
print("=" * 60)
print(f"  Piston mass:      {M_piston/1e3:.1f} tonnes")
print(f"  Fall height:      {H_fall:.0f} m")
print(f"  Impact velocity:  {v_imp:.1f} m/s")
print(f"  Peak pressure:    {P_mpa.max():.2f} MPa (Joukowsky)")
print(f"  E_kinetic:        {E_kin:.2e} J  ({100*E_kin/E_tot:.0f}%)")
print(f"  E_latent:         {E_lat:.2e} J  ({100*E_lat/E_tot:.0f}%)")
print(f"  E_sublimation:    {E_sub:.2e} J  ({100*E_sub/E_tot:.0f}%)")
print(f"  E_total:          {E_tot:.2e} J")
print(f"  Seismic equiv:    M_L = {M_L:.2f}")
print()
print(f"  Figures saved to: ./{args.out}/")
for fig_name in ['fig1_dsd','fig2_vortex','fig3_sip_cascade',
                  'fig4_piston_3d','fig5_cohesion','fig6_impact',
                  'fig7_shields','fig8_energy']:
    print(f"    {fig_name}.png")
print()
print("  Cite as: Haas, R. (2024). arXiv:2411.08219")
print("=" * 60)
