import re

from .config import on_rtd, DOWNLOAD_TIMEOUT
from .logger import getLogger

if not on_rtd:
    import requests
    from astropy.coordinates import SkyCoord

NED_EXTINCTION_URL = "https://ned.ipac.caltech.edu/cgi-bin/nph-calc"
IRSA_DUST_URL = "https://irsa.ipac.caltech.edu/cgi-bin/DUST/nph-dust"

# A_V / E(B-V)_SFD for Landolt V with R_V = 3.1 (Schlafly & Finkbeiner 2011, Table 6);
# this is the calibration NED reports as "Landolt V" in its S&F column.
SF11_AV_PER_EBV_SFD = 2.742


def _ned_AV(coords, timeout):
    rah, ram, ras = coords.ra.hms
    decd, decm, decs = coords.dec.dms
    sign = "+" if coords.dec.deg >= 0 else "-"
    params = {
        "in_csys": "Equatorial",
        "in_equinox": "J2000.0",
        "obs_epoch": "2010",
        "lon": "{:d}:{:d}:{:05.2f}".format(int(rah), int(ram), ras),
        "lat": "{}{:d}:{:d}:{:05.2f}".format(sign, int(abs(decd)), int(abs(decm)), abs(decs)),
        "pa": "0.0",
        "out_csys": "Equatorial",
        "out_equinox": "J2000.0",
    }
    r = requests.get(NED_EXTINCTION_URL, params=params, timeout=timeout)
    r.raise_for_status()
    m = re.search(r"Landolt V \(0\.54\)\s+(\d+\.\d+)", r.text)
    if not m:
        raise RuntimeError("Could not parse NED extinction calculator output ({}).".format(r.url))
    return float(m.group(1))


def _irsa_AV(coords, timeout):
    locstr = "{:.6f} {:.6f} equ j2000".format(coords.ra.deg, coords.dec.deg)
    r = requests.get(IRSA_DUST_URL, params={"locstr": locstr}, timeout=timeout)
    r.raise_for_status()
    m = re.search(r"<refPixelValueSFD>\s*([0-9.]+)", r.text)
    if not m:
        raise RuntimeError("Could not parse IRSA dust service output ({}).".format(r.url))
    return SF11_AV_PER_EBV_SFD * float(m.group(1))


def get_AV_infinity(ra, dec, frame="icrs", timeout=None):
    """
    Gets the A_V exctinction at infinity for a given line of sight.

    Queries the NED extinction calculator (Schlafly & Finkbeiner 2011
    recalibration of the SFD maps, Landolt V).  If NED cannot be reached,
    falls back to the IRSA dust service, converting E(B-V)_SFD at the
    reference pixel with the same S&F11 coefficient (agrees with NED to a
    few per cent).

    :param ra,dec:
        Desired coordinates, in degrees.
    :param frame: (optional)
        Frame of input coordinates (e.g., ``'icrs', 'galactic'``)
    """
    if timeout is None:
        timeout = DOWNLOAD_TIMEOUT
    coords = SkyCoord(ra, dec, unit="deg", frame=frame).transform_to("icrs")

    errors = []
    for name, fn in (("NED", _ned_AV), ("IRSA", _irsa_AV)):
        try:
            return fn(coords, timeout)
        except Exception as e:
            getLogger().warning("A_V query to {} failed: {}".format(name, e))
            errors.append("{}: {}".format(name, e))

    raise RuntimeError("AV query fails! ({})".format("; ".join(errors)))
