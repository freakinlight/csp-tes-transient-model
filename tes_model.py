import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from math import pi, sqrt
from scipy.integrate import solve_ivp

# -------------------------
# 1) Given data
# -------------------------
@dataclass
class Props:
    # Rocks (limestone)
    rho_rock: float = 2750.0          # kg/m3
    cp_rock: float = 840.0            # J/kg.K
    k_rock: float = 1.3               # W/m.K

    # PCM (KCl-LiCl)
    rho_pcm: float = 1650.0           # kg/m3
    cp_pcm: float = 1200.0            # J/kg.K (solid and liquid)
    k_pcm: float = 0.9                # W/m.K
    Tm_pcm_C: float = 355.0           # °C
    Lf_pcm: float = 236e3             # J/kg  (236 kJ/kg)

    # HTF (Syltherm 800)
    rho_htf: float = 600.0            # kg/m3
    cp_htf: float = 2000.0            # J/kg.K
    k_htf: float = 0.07               # W/m.K
    mu_htf: float = 3.1e-4            # Pa.s


@dataclass
class Geo:
    N_tubes: int = 100
    D_tube: float = 0.01              # m (assume inner diameter for convection)
    L: float = 5.0                    # m (height/length)


@dataclass
class BC:
    T_init_C: float = 400.0           # initial temperature of entire system (°C)
    T_in_C: float = 200.0             # HTF inlet temperature (°C)
    T_amb_C: float = 30.0             # ambient temperature (°C)
    h_amb: float = 5.0                # W/m2.K


@dataclass
class Design:
    Q_req: float = 1e6                # W
    t_req_h: float = 4.0              # hours (energy requirement)
    t_sim_h: float = 6.0              # hours (simulation duration)


@dataclass
class ModelChoices:
    include_losses: bool = True
    # Closure: use U = hi (dominant tube-side convection) then scale by UA_factor
    sigma_pcm_K: float = 2.0          # PCM Cp_eff smoothing width (K)


# -------------------------
# 2) Helper functions
# -------------------------
def compute_required_energy(des: Design) -> float:
    return des.Q_req * (des.t_req_h * 3600.0)

def mdot_from_power(des: Design, props: Props, Tin_C: float, Tout_C: float) -> float:
    dT = (Tout_C - Tin_C)
    return des.Q_req / (props.cp_htf * dT)

def tube_bundle_area(g: Geo) -> float:
    return g.N_tubes * pi * g.D_tube * g.L

def tube_flow_area_total(g: Geo) -> float:
    Ac = pi * (g.D_tube**2) / 4.0
    return g.N_tubes * Ac

def reynolds(mdot: float, props: Props, g: Geo) -> float:
    A_total = tube_flow_area_total(g)
    u = mdot / (props.rho_htf * A_total)
    return props.rho_htf * u * g.D_tube / props.mu_htf

def prandtl(props: Props) -> float:
    return props.cp_htf * props.mu_htf / props.k_htf

def hi_from_correlation(mdot: float, props: Props, g: Geo) -> float:
    # h = (k_HTF / D) * [ 4.66 if Re <= 2300 else 0.023 Re^0.8 Pr^0.4 ]
    Re = reynolds(mdot, props, g)
    Pr = prandtl(props)
    if Re <= 2300.0:
        Nu = 4.66
    else:
        Nu = 0.023 * (Re**0.8) * (Pr**0.4)
    return props.k_htf / g.D_tube * Nu

def tank_diameter_from_volume(V: float, L: float) -> float:
    return sqrt(4.0 * V / (pi * L))

def tank_external_area(D_tank: float, L: float) -> float:
    A_lat = pi * D_tank * L
    A_ends = 2.0 * pi * (D_tank**2) / 4.0
    return A_lat + A_ends

def pcm_effective_cp(T_C: float, props: Props, sigma_K: float) -> float:
    """
    Physically consistent effective heat capacity method:
    Cp_eff(T) = Cp_base + Lf * gaussian(T; Tm, sigma)  (integral = Lf)
    """
    Tm = props.Tm_pcm_C
    gauss = (1.0 / (np.sqrt(2.0*np.pi) * sigma_K)) * np.exp(-0.5 * ((T_C - Tm)/sigma_K)**2)
    return props.cp_pcm + props.Lf_pcm * gauss

def size_storage(props: Props, des: Design):
    """
    Uses the Part I temperature window from statement (Ti=290°C, Tf=400°C)
    and energy requirement E = 1 MW * 4 h.
    """
    Ereq = compute_required_energy(des)
    Ti = 290.0
    Tf = 400.0
    dT = Tf - Ti

    m_rock = Ereq / (props.cp_rock * dT)
    V_rock = m_rock / props.rho_rock

    # PCM: stored energy per kg = cp*(Tf-Ti) + Lf
    e_per_kg = props.cp_pcm * dT + props.Lf_pcm
    m_pcm = Ereq / e_per_kg
    V_pcm = m_pcm / props.rho_pcm

    return dict(Ereq=Ereq, m_rock=m_rock, V_rock=V_rock, m_pcm=m_pcm, V_pcm=V_pcm)


# -------------------------
# 3) Simulation
# -------------------------
def simulate_case(case: str, props: Props, geo: Geo, bc: BC, des: Design, choices: ModelChoices,
                  V_storage: float, m_storage: float, UA_factor: float):

    # mdot from required 1 MW with Tin=200C, Tout_target=400C
    mdot = mdot_from_power(des, props, bc.T_in_C, 400.0)

    # Base UA from internal correlation; then apply UA_factor as "enhancement"
    hi = hi_from_correlation(mdot, props, geo)
    A = tube_bundle_area(geo)
    UA_base = hi * A
    UA = UA_factor * UA_base

    # HTF mass inside tubes
    Ac_single = pi * geo.D_tube**2 / 4.0
    V_tubes = geo.N_tubes * Ac_single * geo.L
    m_htf_inside = props.rho_htf * V_tubes

    # Tank geometry for ambient losses (derived from volume + height)
    D_tank = tank_diameter_from_volume(V_storage, geo.L)
    A_ext = tank_external_area(D_tank, geo.L)

    Tin = bc.T_in_C
    Tamb = bc.T_amb_C

    # Initial conditions
    Tf0 = bc.T_init_C
    Ts0 = bc.T_init_C

    t0 = 0.0
    tf = des.t_sim_h * 3600.0

    def rhs(t, y):
        Tf, Ts = y

        # Heat exchange HTF <-> storage
        Qfs = UA * (Tf - Ts)  # W

        # Ambient losses from storage
        Qloss = 0.0
        if choices.include_losses:
            Qloss = bc.h_amb * A_ext * (Ts - Tamb)

        # HTF (lumped) energy balance
        dTfdt = (mdot * props.cp_htf * (Tin - Tf) - Qfs) / (m_htf_inside * props.cp_htf)

        # Storage energy balance
        if case.lower() == "rocks":
            C = m_storage * props.cp_rock
        elif case.lower() == "pcm":
            Cp_eff = pcm_effective_cp(Ts, props, choices.sigma_pcm_K)
            C = m_storage * Cp_eff
        else:
            raise ValueError("case must be 'rocks' or 'pcm'")

        dTsdt = (Qfs - Qloss) / C
        return [dTfdt, dTsdt]

    sol = solve_ivp(rhs, (t0, tf), [Tf0, Ts0], method="RK45", max_step=10.0)

    t = sol.t
    Tf = sol.y[0]
    Ts = sol.y[1]

    # Outputs
    Tout = Tf
    Qfs = UA * (Tf - Ts)        # W ; negative means heat goes from storage to HTF
    Qreleased = -Qfs            # W ; positive delivered to HTF during discharge

    meta = dict(
        mdot=mdot,
        Re=reynolds(mdot, props, geo),
        hi=hi,
        A=A,
        UA_base=UA_base,
        UA=UA,
        UA_factor=UA_factor,
        D_tank=D_tank,
        A_ext=A_ext
    )
    return t, Tout, Qreleased, Ts, meta


def main():
    props = Props()
    geo = Geo()
    bc = BC()
    des = Design()
    choices = ModelChoices(include_losses=True, sigma_pcm_K=2.0)

    sizing = size_storage(props, des)

    # ---- Sensitivity settings ----
    UA_factors = [1.0, 1.5, 2.0, 3.0]   # you can change these
    t_plot_start_h = 0.05              # crop first 3 minutes to avoid initial spike

    results = {"rocks": [], "pcm": []}

    for f in UA_factors:
        t_r, Tout_r, Q_r, Ts_r, meta_r = simulate_case(
            "rocks", props, geo, bc, des, choices,
            V_storage=sizing["V_rock"], m_storage=sizing["m_rock"], UA_factor=f
        )
        t_p, Tout_p, Q_p, Ts_p, meta_p = simulate_case(
            "pcm", props, geo, bc, des, choices,
            V_storage=sizing["V_pcm"], m_storage=sizing["m_pcm"], UA_factor=f
        )

        results["rocks"].append((t_r, Tout_r, Q_r, Ts_r, meta_r))
        results["pcm"].append((t_p, Tout_p, Q_p, Ts_p, meta_p))

    # ---- Plot Tout(t) ----
    plt.figure()
    for (t, Tout, Q, Ts, meta) in results["rocks"]:
        th = t / 3600.0
        mask = th >= t_plot_start_h
        plt.plot(th[mask], Tout[mask], label=f"Rocks, UA×{meta['UA_factor']:.1f}")
    for (t, Tout, Q, Ts, meta) in results["pcm"]:
        th = t / 3600.0
        mask = th >= t_plot_start_h
        plt.plot(th[mask], Tout[mask], linestyle="--", label=f"PCM, UA×{meta['UA_factor']:.1f}")
    plt.xlabel("Time (h)")
    plt.ylabel("HTF outlet temperature $T_{out}$ (°C)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("Tout_vs_time_UA_sensitivity.png", dpi=200)

    # ---- Plot Qdot(t) ----
    plt.figure()
    # 1 MW reference line
    plt.axhline(des.Q_req / 1e6, linewidth=1.0, label="1 MW target")

    for (t, Tout, Q, Ts, meta) in results["rocks"]:
        th = t / 3600.0
        mask = th >= t_plot_start_h
        plt.plot(th[mask], Q[mask] / 1e6, label=f"Rocks, UA×{meta['UA_factor']:.1f}")
    for (t, Tout, Q, Ts, meta) in results["pcm"]:
        th = t / 3600.0
        mask = th >= t_plot_start_h
        plt.plot(th[mask], Q[mask] / 1e6, linestyle="--", label=f"PCM, UA×{meta['UA_factor']:.1f}")

    plt.xlabel("Time (h)")
    plt.ylabel("Released thermal power $\\dot{Q}$ (MW)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("Qdot_vs_time_UA_sensitivity.png", dpi=200)

    # ---- Print summary ----
    print("=== DESIGN SIZING (1 MW for 4 h) ===")
    print(f"Ereq  = {sizing['Ereq']:.3e} J (= 4 MWh)")
    print(f"Rocks: m = {sizing['m_rock']:.3e} kg, V = {sizing['V_rock']:.3f} m^3")
    print(f"PCM  : m = {sizing['m_pcm']:.3e} kg, V = {sizing['V_pcm']:.3f} m^3\n")

    # Show baseline transfer info (UA×1)
    base_meta = results["rocks"][0][4]
    print("=== BASELINE HEAT TRANSFER (UA×1.0) ===")
    print(f"mdot    = {base_meta['mdot']:.4f} kg/s")
    print(f"Re      = {base_meta['Re']:.2e}")
    print(f"hi      = {base_meta['hi']:.2f} W/m^2.K")
    print(f"A       = {base_meta['A']:.2f} m^2")
    print(f"UA_base = {base_meta['UA_base']:.2f} W/K\n")

    print("Saved figures:")
    print(" - Tout_vs_time_UA_sensitivity.png")
    print(" - Qdot_vs_time_UA_sensitivity.png")


if __name__ == "__main__":
    main()

