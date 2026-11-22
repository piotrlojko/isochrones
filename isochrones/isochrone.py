from .config import on_rtd

from .logger import getLogger

if not on_rtd:
    from astropy import constants as const

    # Define useful constants
    G = const.G.cgs.value
    MSUN = const.M_sun.cgs.value
    RSUN = const.R_sun.cgs.value

    from .models import ModelGridInterpolator

else:
    G = 6.67e-11
    MSUN = 1.99e33
    RSUN = 6.96e10

from .config import ISOCHRONES


def get_ichrone(models, bands=None, default=False, tracks=False, basic=False, **kwargs):
    """Gets Isochrone Object by name, or type, with the right bands

    If `default` is `True`, then will set bands
    to be the union of bands and default_bands
    """
    if not bands:
        bands = None

    if isinstance(models, ModelGridInterpolator):
        return models

    if isinstance(models, type):
        ichrone = models(bands)
    elif models == "mist":
        if tracks:
            if basic:
                from isochrones.mist import MIST_BasicEvolutionTrack

                ichrone = MIST_BasicEvolutionTrack(bands=bands, **kwargs)
            else:
                from isochrones.mist import MIST_EvolutionTrack

                ichrone = MIST_EvolutionTrack(bands=bands, **kwargs)
        else:
            if basic:
                from isochrones.mist import MIST_BasicIsochrone

                ichrone = MIST_BasicIsochrone(bands=bands, **kwargs)
            else:
                from isochrones.mist import MIST_Isochrone

                ichrone = MIST_Isochrone(bands=bands, **kwargs)
    else:
        raise ValueError("Unknown stellar models: {}".format(models))
    return ichrone
