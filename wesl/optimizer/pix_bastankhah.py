import jax.numpy as jnp

from pixwake import WakeSimulation
from pixwake.deficit import BastankhahGaussianDeficit
from pixwake.superposition import LinearSum
from pixwake.utils import ct2a_mom1d


class PixBastankhahGaussianDeficit(WakeSimulation):
    def __init__(self, site, turbines, k, **kwargs):
        deficit = BastankhahGaussianDeficit(k, use_effective_ws=True, ct2a=ct2a_mom1d)  # , superposition=LinearSum())
        super().__init__(turbines, deficit, **kwargs)

        self.wd, self.ws = jnp.meshgrid(*site.get_defaults())
        self.ws = self.ws.flatten()
        self.wd = self.wd.flatten()

    def __call__(self, x, y, *args, **kwargs):
        list_args = list(args)

        try:
            ws_amb = list_args.pop(0)
        except IndexError:
            ws_amb = jnp.asarray(self.ws)

        try:
            wd_amb = list_args.pop(0)
        except IndexError:
            wd_amb = jnp.asarray(self.wd)

        args = tuple(list_args)

        return super().__call__(jnp.asarray(x), jnp.asarray(y), ws_amb, wd_amb, *args, **kwargs)