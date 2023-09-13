from packed_bed import PackedBedModel
from matplotlib import pyplot as plt

D = 3.5
t_i = 0.20  # Insulation thickness [m]
t_g = 0.20  # Ground thickness [m]

P = 30e6  # Maximum working pressure
t_steel = P * (D + 2 * t_i) / (2 * (140e6 - 0.6 * P))  # Steel thickness [m]

pbed = PackedBedModel(
    T_d=500 + 273,
    P=27.5e6,
    L=9.1,
    D=3.5,
    d=0.003,
    eps=0.35,
    rho_s=3950,
    T_env=10 + 273,
    t_wall=[t_i, t_steel, t_g],
    k_wall=[0.25, 11.7, 2.9],
    rho_wall=[250, 8050, 2650],
    cp_wall=[1190, 483.1, 732.2]
)

fig, ax = plt.subplots(figsize=(4, 6))
pbed.plot_temperature_contour(ax)

plt.tight_layout()
plt.show()
