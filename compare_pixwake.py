import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from py_wake.deficit_models.gaussian import BastankhahGaussianDeficit
from py_wake.literature.gaussian_models import Bastankhah_PorteAgel_2014
from py_wake.superposition_models import SquaredSum
from py_wake.wind_farm_models.engineering_models import All2AllIterative
from py_wake.wind_turbines import WindTurbines
from py_wake.wind_turbines.power_ct_functions import PowerCtTabular

from pixwake import Curve, Turbine
from pixwake.deficit.gaussian import BastankhahGaussianDeficit as BastankhahGaussianDeficit_PIX
from pixwake.plot import plot_flow_map

from wesl.optimizer.pix_bastankhah import PixBastankhahGaussianDeficit
from wesl.offshore_wind_farms.vineyard_wind import VineyardWind, x_vineyard, y_vineyard


def _create_pywake_turbines(n_turbines, ct_curve, power_curve, RD=120.0, HH=100.0):
    names = [f"WT{i}" for i in range(n_turbines)]
    power_values_W = power_curve[:, 1] * 1000
    power_ct_func = PowerCtTabular(
        ws=power_curve[:, 0],
        power=power_values_W,
        power_unit="w",
        ct=ct_curve[:, 1],
    )
    return WindTurbines(
        names=names,
        diameters=[RD] * n_turbines,
        hub_heights=[HH] * n_turbines,
        powerCtFunctions=[power_ct_func] * n_turbines,
    )


def _create_pixwake_turbine(ct_curve, power_curve, RD=120.0, HH=100.0):
    return Turbine(
        rotor_diameter=RD,
        hub_height=HH,
        power_curve=Curve(ws=power_curve[:, 0], values=power_curve[:, 1]),
        ct_curve=Curve(ws=ct_curve[:, 0], values=ct_curve[:, 1]),
    )


n_turbines = 1

ct_vals = np.array([
    0.00, 0.00, 0.00, 0.80, 0.79, 0.77, 0.75, 0.72, 0.68, 0.64,
    0.62, 0.61, 0.60, 0.55, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25,
    0.20, 0.18, 0.15, 0.12, 0.10, 0.10,
])

power_vals = np.array([
    0, 0, 0, 100, 300, 600, 1200, 1800, 2300, 2700,
    2900, 2950, 3000, 3000, 3000, 3000, 3000, 3000,
    3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000,
])

cutout_ws = 25.0
ct_pw_ws = np.arange(0.0, cutout_ws + 1.0, 1.0)
ct_curve = np.stack([ct_pw_ws, ct_vals], axis=1)
power_curve = np.stack([ct_pw_ws, power_vals], axis=1)

site = VineyardWind()
x, y = x_vineyard[:n_turbines], y_vineyard[:n_turbines]

# PyWake
windTurbines = _create_pywake_turbines(n_turbines=n_turbines, ct_curve=ct_curve, power_curve=power_curve)

wake_model = BastankhahGaussianDeficit(k=0.0324555)
wfm = All2AllIterative(
        site,
        windTurbines,
        wake_deficitModel=wake_model,
        superpositionModel=SquaredSum(),
    )
sim_res = wfm(x, y)

#change the wind speed and wind direction to visualize different flow cases
wsp = 9
wdir = 90
flow_map = sim_res.flow_map(grid=None, wd=wdir, ws=wsp)

print(f'PyWake AEP: {sim_res.aep().sum()}\n---')

# PixWake
pix_windTurbines = _create_pixwake_turbine(ct_curve=ct_curve, power_curve=power_curve)

pix_wf_model = PixBastankhahGaussianDeficit(site, pix_windTurbines, k=0.0324555)
pix_flow_map, (pix_flowmap_x, pix_flowmap_y) = pix_wf_model.flow_map(x, y, ws=wsp, wd=wdir)

print(f'PixWake AEP: {pix_wf_model(x, y).aep()}\n---')

fig, axs = plt.subplots(1, 2, figsize=(10, 10))
fig.suptitle('Flowmap for'+ f' {wdir} deg and {wsp} m/s')

flow_map.plot_wake_map(ax = axs[0], cmap='viridis')
axs[0].set_xlabel('x [m]')
axs[0].set_ylabel('y [m]')
axs[0].set_title('PyWake')
axs[0].set_aspect("equal", adjustable='box')

plot_flow_map(pix_flowmap_x, pix_flowmap_y, pix_flow_map, wt_x=jnp.asarray(x), wt_y=jnp.asarray(y), ax=axs[1])
axs[1].set_xlabel('x [m]')
axs[1].set_ylabel('y [m]')
axs[1].set_title('PixWake')

plt.show()
