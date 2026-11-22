import os
import re
import time

import numpy as np
import pandas as pd
import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.vizier import VizierClass

from ..config import VIZIER_MIRRORS, DOWNLOAD_TIMEOUT
from ..logger import getLogger
from .query import EmptyQueryError
from .catalog import Catalog

__all__ = [
    "MirroredVizier",
    "VizierUnavailableError",
    "get_vizier_mirrors",
    "query_region_with_mirrors",
    "vizier_mirror_status",
    "VizierCatalog",
    "TwoMASS",
    "Tycho2",
    "WISE",
    "Gaia",
]


class VizierUnavailableError(ConnectionError):
    """Raised when a query failed on every configured VizieR mirror."""


class MirroredVizier(VizierClass):
    """``astroquery`` Vizier client that also accepts full ``http(s)://`` server URLs.

    astroquery always talks https to ``vizier_server``; some official VizieR
    mirrors (NAOJ, China-VO) only answer over plain http, so a server given as
    ``"http://host"`` is used verbatim.
    """

    def _server_to_url(self, return_type="votable"):
        server = self.VIZIER_SERVER
        if server.startswith(("http://", "https://")):
            return server.rstrip("/") + "/viz-bin/" + return_type
        return super()._server_to_url(return_type=return_type)


# VizieR reports server-side problems (e.g. "Table or Catalog not found" on a
# mirror lacking a catalogue, database errors) inside an otherwise valid
# VOTable, which astroquery turns into an empty result.  Such responses must
# not be mistaken for a genuinely empty field.
_VIZIER_ERROR_RE = re.compile(
    r'<INFO[^>]*\bname="(?:Error|QUERY_STATUS)"[^>]*\bvalue="(?!(?:OK|OVERFLOW)")([^"]*)"', re.IGNORECASE
)

# Mirrors that failed at the network level are moved to the back of the queue
# for the rest of the session (so a dead primary does not cost a timeout on
# every query).
_dead_mirrors = {}
_DEAD_MIRROR_TTL = 600.0  # seconds


def _server_key(server):
    return re.sub(r"^https?://", "", server).rstrip("/")


def get_vizier_mirrors():
    """Ordered list of VizieR servers to try.

    Starts from :data:`isochrones.config.VIZIER_MIRRORS` (environment variable
    ``ISOCHRONES_VIZIER_MIRRORS``); a server explicitly configured for
    astroquery (``[vizier] server`` in ``astroquery.cfg``) is tried first.
    Mirrors that recently failed are moved to the end.
    """
    mirrors = list(VIZIER_MIRRORS)
    try:
        if os.getenv("ISOCHRONES_VIZIER_MIRRORS"):
            raise LookupError  # explicit isochrones setting wins
        from astroquery.vizier import conf

        configured = conf.server
        if isinstance(configured, str) and configured and _server_key(configured) not in map(_server_key, mirrors):
            mirrors.insert(0, configured)
        elif isinstance(configured, str) and configured and _server_key(configured) != _server_key(mirrors[0]):
            # User explicitly prefers one of the known mirrors.
            idx = [_server_key(m) for m in mirrors].index(_server_key(configured))
            mirrors.insert(0, mirrors.pop(idx))
    except Exception:
        pass

    now = time.time()
    alive = [m for m in mirrors if now - _dead_mirrors.get(_server_key(m), -np.inf) > _DEAD_MIRROR_TTL]
    dead = [m for m in mirrors if m not in alive]
    return alive + dead


def _mark_dead(server):
    _dead_mirrors[_server_key(server)] = time.time()


def query_region_with_mirrors(coords, radius, catalog, columns=("**",), mirrors=None, timeout=None, **kwargs):
    """Run a VizieR cone search, falling back through the mirror list.

    Returns ``(table_list, server)``.  A mirror is skipped if it cannot be
    reached, returns an HTTP error, or reports a server-side error (e.g. the
    catalogue is not available there).  A clean empty answer is returned as is.
    """
    if mirrors is None:
        mirrors = get_vizier_mirrors()
    if timeout is None:
        timeout = DOWNLOAD_TIMEOUT

    log = getLogger()
    errors = []
    for server in mirrors:
        v = MirroredVizier(columns=list(columns), vizier_server=server, timeout=timeout, **kwargs)
        try:
            response = v.query_region_async(coords, radius=radius, catalog=catalog, cache=False)
            response.raise_for_status()
        except Exception as e:
            log.warning("VizieR mirror {} failed ({}); trying next mirror.".format(server, e))
            errors.append("{}: {}".format(server, e))
            _mark_dead(server)
            continue

        server_errors = [m for m in _VIZIER_ERROR_RE.findall(response.text) if m.strip()]
        if server_errors:
            msg = server_errors[0].strip()
            log.warning("VizieR mirror {} reported an error for {} ({}); trying next mirror.".format(server, catalog, msg))
            errors.append("{}: {}".format(server, msg))
            continue

        try:
            result = v._parse_result(response)
        except Exception as e:
            log.warning("Could not parse response from VizieR mirror {} ({}); trying next mirror.".format(server, e))
            errors.append("{}: {}".format(server, e))
            continue

        return result, server

    raise VizierUnavailableError(
        "VizieR query for {} failed on all mirrors:\n  {}".format(catalog, "\n  ".join(errors))
    )


def vizier_mirror_status(catalog="2mass", timeout=20):
    """Check which configured VizieR mirrors currently answer queries.

    Returns a :class:`pandas.DataFrame` with one row per mirror (``ok``,
    response ``time`` in seconds, and ``error`` message).
    """
    coords = SkyCoord(45.0343, 0.2354, unit="deg")
    rows = []
    for server in get_vizier_mirrors():
        t0 = time.time()
        try:
            query_region_with_mirrors(coords, 5 * u.arcsec, catalog, columns=("_r",), mirrors=[server], timeout=timeout)
            rows.append(dict(server=server, ok=True, time=time.time() - t0, error=""))
        except Exception as e:
            rows.append(dict(server=server, ok=False, time=time.time() - t0, error=str(e).splitlines()[-1]))
    return pd.DataFrame(rows)


class VizierCatalog(Catalog):
    columns = ("**", "_r", "_RAJ2000", "_DEJ2000")

    #: Server that answered the last query (set after the query runs).
    server = None

    def _run_query(self):
        if self._empty:
            raise EmptyQueryError("{} is empty!".format(self))

        result, self.server = query_region_with_mirrors(
            self.query_coords, self.query.radius, self.vizier_name, columns=self.columns
        )
        if len(result) == 0:
            self._empty = True
            raise EmptyQueryError("{} returns empty!".format(self))

        self._table = result[0]
        self._coords = SkyCoord(self._table["_RAJ2000"], self._table["_DEJ2000"], unit="deg")
        self._table["PA"] = self.coords.position_angle(self.query_coords).deg
        # self._table['separation'] = self.coords.separation(self.query_coords).arcsec


class TwoMASS(VizierCatalog):
    name = "twomass"
    vizier_name = "2mass"
    epoch = 2000.0
    bands = {"Jmag": "J", "Hmag": "H", "Kmag": "K"}
    id_column = ("2MASS", "_2MASS")


class Tycho2(VizierCatalog):
    name = "Tycho2"
    vizier_name = "tycho2"
    epoch = 2000
    bands = {"BTmag": "BT", "VTmag": "VT"}
    conversions = ["B", "V"]

    def get_id(self, brightest=False):
        if brightest:
            row = self.brightest
        else:
            row = self.closest

        return "{:.0f}-{:.0f}-{:.0f}".format(row["TYC1"], row["TYC2"], row["TYC3"])

    def V(self, brightest=False):
        """
        http://www.aerith.net/astro/color_conversion.html
        """
        mags = self.get_photometry(brightest=brightest, convert=False)
        VT, dVT = mags["VT"]
        BT, dBT = mags["BT"]
        if -0.25 < BT - VT < 2.0:
            (a, b, c, d) = (0.00097, 0.1334, 0.05486, 0.01998)
            V = VT + a - b * (BT - VT) + c * (BT - VT) ** 2 - d * (BT - VT) ** 3

            dVdVT = 1 + b - 2 * c * (BT - VT) + 3 * d * (BT - VT) ** 2
            dVdBT = -b + 2 * c * (BT - VT) - 3 * d * (BT - VT) ** 2
            dV = np.sqrt((dVdVT ** 2 * dVT ** 2) + (dVdBT ** 2 * dBT ** 2))

        else:
            raise ValueError("BT-VT outside of range to convert")

        return V, dV

    def BmV(self, brightest=False):
        mags = self.get_photometry(brightest=brightest, convert=False)
        VT, dVT = mags["VT"]
        BT, dBT = mags["BT"]
        if 0.5 < (BT - VT) < 2.0:
            (e, f, g) = (0.007813, 0.1489, 0.03384)
            BmV = (BT - VT) - e * (BT - VT) - f * (BT - VT) ** 2 + g * (BT - VT) ** 3

            dBmVdVT = -1 + e + 2 * f * (BT - VT) - 3 * g * (BT - VT) ** 2
            dBmVdBT = -dBmVdVT

        elif -0.25 < (BT - VT) < 0.5:
            (h, i, j) = (0.006, 0.1069, 0.1459)
            BmV = (BT - VT) - h - i * (BT - VT) + j * (BT - VT) ** 2

            dBmVdVT = -1 - i - 2 * j * (BT - VT)
            dBmVdBT = -dBmVdVT

        else:
            raise ValueError("BT-VT outside of range to convert")

        dBmV = np.sqrt((dBmVdVT ** 2 * dVT ** 2) + (dBmVdBT ** 2 * dBT ** 2))
        return BmV, dBmV

    def B(self, brightest=False):
        BmV, dBmV = self.BmV(brightest=brightest)
        V, dV = self.V(brightest=brightest)

        B = BmV + V
        dB = np.sqrt(dBmV ** 2 + dV ** 2)
        return B, dB


class WISE(VizierCatalog):
    name = "WISE"
    vizier_name = "allwise"
    epoch = 2000
    bands = {"W1mag": "W1", "W2mag": "W2", "W3mag": "W3"}  # W4 left out.
    id_column = "AllWISE"


class Gaia(VizierCatalog):
    name = "Gaia"
    vizier_name = "I/345/gaia2"
    epoch = 2015.5
    bands = {"Gmag": "G", "BPmag": "BP", "RPmag": "RP"}
    id_column = "Source"

    @property
    def is_good(self):
        """Quality cut from Appendix B of https://arxiv.org/pdf/1804.09378.pdf
        """
        t = self._table
        good = t["RPlx"] > 10
        good &= t["RFG"] > 50
        good &= t["RFRP"] > 20
        good &= t["RFBP"] > 20

        # Dropping the following, because the stars I need always fail...?
        # good &= t['E_BP-RP_'] < 1.3 + 0.06 * (t['BPmag'] - t['RPmag'])**2
        # good &= t['E_BP-RP_'] > 1.0 + 0.015 * (t['BPmag'] - t['RPmag'])**2

        good &= t["Nper"] > 8

        factor = np.where(np.exp(-0.4 * (t["Gmag"] - 19.5)) < 1, 1.0, np.exp(-0.4 * (t["Gmag"] - 19.5)))
        good &= t["chi2AL"] / (t["NgAL"] - 5) < 1.44 * factor

        # Masked entries (missing values) fail the cut.
        good = np.ma.filled(np.ma.asarray(good), False).astype(bool)
        return pd.Series(good, dtype=bool)
