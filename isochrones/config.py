"""Package-wide configuration.

Everything here can be overridden with environment variables, which is the
recommended way to point isochrones at different data directories or at
alternative mirrors of the remote services it depends on:

``ISOCHRONES``
    Directory where model grids and derived caches are stored
    (default: ``~/.isochrones``).

``ISOCHRONES_MIST_MIRRORS``
    Comma-separated list of base URLs hosting the MIST data products, tried
    in order.  Each base URL must expose the same directory layout as
    https://mist.science/ (``data/tarballs_v1.2/...``, ``BC_tables/...``).

``ISOCHRONES_VIZIER_MIRRORS``
    Comma-separated list of VizieR servers, tried in order.  Entries may be
    bare host names (queried over https) or full ``http://``/``https://``
    URLs (needed for mirrors that do not serve https).

``ISOCHRONES_DOWNLOAD_TIMEOUT``
    Network timeout (seconds) for data downloads (default: 60).

``ISOCHRONES_DOWNLOAD_RETRIES``
    Number of attempts per URL before moving to the next mirror (default: 3).
"""

import os

on_rtd = os.environ.get("READTHEDOCS") == "True"

ISOCHRONES = os.getenv("ISOCHRONES", os.path.expanduser(os.path.join("~", ".isochrones")))

POLYCHORD = os.getenv("POLYCHORD", os.path.expanduser(os.path.join("~", "PolyChord")))


def _env_list(name, default):
    value = os.getenv(name)
    if not value:
        return tuple(default)
    return tuple(v.strip() for v in value.split(",") if v.strip())


# MIST moved from http://waps.cfa.harvard.edu/MIST to https://mist.science in
# late 2025 (the old address now redirects there).  The Internet Archive is
# kept as a last resort: it holds snapshots of the original BC tables and of
# some of the model tarballs, which are validated after download.
DEFAULT_MIST_MIRRORS = (
    "https://mist.science",
    "http://waps.cfa.harvard.edu/MIST",
    "https://web.archive.org/web/2025id_/http://waps.cfa.harvard.edu/MIST",
)
MIST_MIRRORS = _env_list("ISOCHRONES_MIST_MIRRORS", DEFAULT_MIST_MIRRORS)

# Primary CDS VizieR server followed by the official mirrors listed at
# https://cds.unistra.fr/ (NAOJ and China-VO only answer over plain http).
DEFAULT_VIZIER_MIRRORS = (
    "vizier.cds.unistra.fr",
    "vizier.cfa.harvard.edu",
    "vizier.idia.ac.za",
    "http://vizier.nao.ac.jp",
    "http://vizier.china-vo.org",
)
VIZIER_MIRRORS = _env_list("ISOCHRONES_VIZIER_MIRRORS", DEFAULT_VIZIER_MIRRORS)

DOWNLOAD_TIMEOUT = float(os.getenv("ISOCHRONES_DOWNLOAD_TIMEOUT", 60))
DOWNLOAD_RETRIES = int(os.getenv("ISOCHRONES_DOWNLOAD_RETRIES", 3))
