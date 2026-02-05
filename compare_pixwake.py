from time import process_time

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from py_wake.deficit_models.gaussian import BastankhahGaussianDeficit
from py_wake.literature.gaussian_models import Bastankhah_PorteAgel_2014
from py_wake.superposition_models import SquaredSum
from py_wake.wind_farm_models.engineering_models import All2AllIterative
from py_wake.wind_turbines import WindTurbines
from py_wake.wind_turbines.power_ct_functions import PowerCtTabular

from pixwake import Curve, Turbine, WakeSimulation
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


n_turbines = 200

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

wake_expansion_k = 0.0324555

cutout_ws = 25.0
ct_pw_ws = np.arange(0.0, cutout_ws + 1.0, 1.0)
ct_curve = np.stack([ct_pw_ws, ct_vals], axis=1)
power_curve = np.stack([ct_pw_ws, power_vals], axis=1)

site = VineyardWind()
x, y = x_vineyard[:n_turbines], y_vineyard[:n_turbines]

# PyWake
windTurbines = _create_pywake_turbines(n_turbines=n_turbines, ct_curve=ct_curve, power_curve=power_curve)

wake_model = BastankhahGaussianDeficit(k=wake_expansion_k)
wfm = All2AllIterative(
        site,
        windTurbines,
        wake_deficitModel=wake_model,
        superpositionModel=SquaredSum(),
    )
wfm = Bastankhah_PorteAgel_2014(
    site,
    windTurbines,
    k=wake_expansion_k,
    superpositionModel=SquaredSum(),
)
sim_res = wfm(x, y)

#change the wind speed and wind direction to visualize different flow cases
wsp = 9
wdir = 270
flow_map = sim_res.flow_map(grid=None, wd=wdir, ws=wsp)

print(f'PyWake AEP: {sim_res.aep().sum().values}')

# PixWake
pix_windTurbines = _create_pixwake_turbine(ct_curve=ct_curve, power_curve=power_curve)

pix_wf_model = PixBastankhahGaussianDeficit(site, pix_windTurbines, k=wake_expansion_k, use_radius_mask=False)
pix_flow_map, (pix_flowmap_x, pix_flowmap_y) = pix_wf_model.flow_map(x, y, ws=wsp, wd=wdir)

print(f'PixWake AEP: {pix_wf_model(x, y).aep()}')

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
axs[1].get_legend().remove()

print(30*'<>')

cutin_ws = 3.0

x, y = x_vineyard[:n_turbines], y_vineyard[:n_turbines]
windTurbines = _create_pywake_turbines(n_turbines, ct_curve, power_curve)

wake_model = BastankhahGaussianDeficit(k=wake_expansion_k)

wfm = All2AllIterative(
    site,
    windTurbines,
    wake_deficitModel=wake_model,
    superpositionModel=SquaredSum(),
)

n_timestamps = 1000
ws, wd = (
    np.random.uniform(cutin_ws, cutout_ws, size=n_timestamps),
    np.random.uniform(0, 360, size=n_timestamps),
)

# Doing this gives different results, don't know why yet though
wd, ws = site.get_defaults()
ws_, wd_ = jnp.meshgrid(ws, wd)
ws_ = ws_.flatten()
wd_ = wd_.flatten()


start_time = process_time()
# sim_res = wfm(x=x, y=y, wd=wd, ws=ws)
sim_res = wfm(x=x, y=y)
print(f"PyWake time: {process_time() - start_time} s")
pywake_ws_eff = sim_res["WS_eff"].values

# bug in PyWake not masking the contributions from far off wake radius
model = BastankhahGaussianDeficit_PIX(k=wake_expansion_k, use_radius_mask=False)
turbine = Turbine(
    rotor_diameter=windTurbines.diameter().item(),
    hub_height=100.0,
    power_curve=Curve(ws=power_curve[:, 0], values=power_curve[:, 1]),
    ct_curve=Curve(ws=ct_curve[:, 0], values=ct_curve[:, 1]),
)
sim = WakeSimulation(turbine, model, fpi_damp=1.0, mapping_strategy="map")
start_time = process_time()
pixwake_sim_res = sim(
    jnp.asarray(x),
    jnp.asarray(y),
    ws_,
    wd_,
)
print(f"PixWake time: {process_time() - start_time} s")
print(15 * '~')

P_ilk = site.local_wind(ws=ws, wd=wd).P_ilk
pix_probs = P_ilk.reshape((1, pixwake_sim_res.effective_ws.shape[0])).T

print(f"PyWake Effective Wind Speed: {np.maximum(pywake_ws_eff, 0).shape}")
print(f"PixWake Effective Wind Speed: {pixwake_sim_res.effective_ws.T.shape}")

print(f"Pywake AEP: {sim_res.aep().sum().values}")
print(f"PixWake AEP: {pixwake_sim_res.aep(probabilities=pix_probs)}")

flow_map_ = sim_res.flow_map(grid=None, wd=wdir, ws=wsp)
pix_flow_map_, (pix_flowmap_x_, pix_flowmap_y_) = sim.flow_map(x, y, ws=wsp, wd=wdir)

fig_, axs_ = plt.subplots(1, 2, figsize=(10, 10))
fig_.suptitle('Flowmap for'+ f' {wdir} deg and {wsp} m/s')

flow_map_.plot_wake_map(ax=axs_[0], cmap='viridis')
axs_[0].set_xlabel('x [m]')
axs_[0].set_ylabel('y [m]')
axs_[0].set_title('PyWake')
axs_[0].set_aspect("equal", adjustable='box')

plot_flow_map(pix_flowmap_x_, pix_flowmap_y_, pix_flow_map_, wt_x=jnp.asarray(x), wt_y=jnp.asarray(y), ax=axs_[1])
axs_[1].set_xlabel('x [m]')
axs_[1].set_ylabel('y [m]')
axs_[1].set_title('PixWake')
axs_[1].get_legend().remove()

plt.show()
