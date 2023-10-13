import packed_bed
from packed_bed import PackedBedModel
from matplotlib import pyplot as plt
import numpy as np

D = 3.5  # m
h_air = 5  # W/m^2 K
t_ins = 102e-3  # m
t_st = 127e-3  # m
k_g = 2.9  # W/m K

R_st = D/2 + t_ins + t_st
t_g = R_st * (np.exp(k_g / (h_air * R_st)) - 1)

pbed = PackedBedModel(
    T_initial=60 + 273,
    P=27.5e6,
    L=9.1,
    D=D,
    d=3e-3,
    eps=0.35,
    rho_s=3950,
    T_env=10 + 273,
    t_wall=[t_ins, t_st, t_g],
    k_wall=[0.25, 11.7, k_g],
    rho_wall=[250, 8050, 2650],
    cp_wall=[1190, 483.1, 732.2],
    n=1200
)

try:
    t_charge = pbed.advance(
        T_inlet=750 + 273,
        T_outlet_stop=725 + 273,
        P_inlet=27.5e6,
        m_dot_inlet=8.17,
        dt=30,
        charge=True,
    )

    plt.figure()
    plt.title("Charging")
    plt.ylabel("Temperature, $T$ [°C]")
    plt.xlabel("Axial position, $z$ [m]")

    for i in range(int(t_charge / 3600) + 1):
        plt.plot(pbed.z, pbed.T_s[pbed.time(h=i)] - 273, label=f"{i}h")

    plt.plot(pbed.z, pbed.T_s[-1] - 273, label=f"{pbed.t[-1] / 3600:.1f}h")

    plt.ylim(0, 800)
    plt.xlim(0, max(pbed.z))
    plt.legend()

    t_discharge = pbed.advance(
        T_inlet=500 + 273,
        T_outlet_stop=650 + 273,
        P_inlet=27.5e6,
        m_dot_inlet=8.17,
        dt=30,
        charge=False,
    )

    plt.figure()
    plt.title("Discharging")
    plt.ylabel("Temperature, $T$ [°C]")
    plt.xlabel("Axial position, $z$ [m]")

    for i in range(int(t_charge / 3600) + 1, int(pbed.t[-1] / 3600) + 1):
        plt.plot(pbed.z, pbed.T_s[pbed.time(h=i)] - 273, label=f"{i}h")

    plt.plot(pbed.z, pbed.T_s[-1] - 273, label=f"{pbed.t[-1] / 3600:.1f}h")

    plt.ylim(450, 750)
    plt.xlim(0, max(pbed.z))
    plt.legend()

    plt.show()

except packed_bed.StopCriterionError:
    print(f"ERROR: Stopped at t = {pbed.t[-1]} s.")
