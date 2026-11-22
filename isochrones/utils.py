import os
import tarfile
import time

import requests
import numpy as np
import numba as nb
from math import log10

from .logger import getLogger
from .version import __version__

USER_AGENT = "isochrones/{} (+https://github.com/timothydmorton/isochrones)".format(__version__)


class DownloadError(IOError):
    """Raised when a file could not be retrieved from any of the candidate URLs."""


def band_pairs(bands):
    return [(bands[i], bands[-1]) for i in range(len(bands) - 1)]


def _as_url_list(url):
    if isinstance(url, str):
        return [url]
    urls = []
    for u in url:
        if u not in urls:
            urls.append(u)
    return urls


def download_file(url, path=None, clobber=False, timeout=None, retries=None, validate=None, chunk_size=1 << 20):
    """Download a file, trying each candidate URL (mirror) in turn.

    Parameters
    ----------
    url : str or sequence of str
        URL, or list of equivalent URLs (e.g. the same file on several mirrors).
        They are tried in order until one succeeds.
    path : str, optional
        Local path to download to.  Defaults to the URL basename in the
        current directory.
    clobber : bool
        Re-download even if ``path`` already exists.
    timeout : float, optional
        Connect/read timeout in seconds
        (default: :data:`isochrones.config.DOWNLOAD_TIMEOUT`).
    retries : int, optional
        Attempts per URL (default: :data:`isochrones.config.DOWNLOAD_RETRIES`).
    validate : callable, optional
        ``validate(path)`` is called on the downloaded file and should raise
        if the content is not what was expected (e.g. an HTML error page or
        a different data product); the file is then discarded and the next
        URL is tried.

    Returns
    -------
    str
        The local filename.

    The data are first written to ``<path>.part`` and only moved into place
    once the transfer completed (and validated), so interrupted downloads
    never leave a truncated file behind.
    """
    from . import config

    urls = _as_url_list(url)
    if timeout is None:
        timeout = config.DOWNLOAD_TIMEOUT
    if retries is None:
        retries = config.DOWNLOAD_RETRIES

    local_filename = path if path is not None else os.path.basename(urls[0].split("?")[0])

    if os.path.exists(local_filename) and not clobber:
        getLogger().info("{} exists; not downloading.".format(local_filename))
        return local_filename

    directory = os.path.dirname(os.path.abspath(local_filename))
    os.makedirs(directory, exist_ok=True)
    tmp_filename = local_filename + ".part"

    try:
        from tqdm.auto import tqdm
    except ImportError:  # pragma: no cover
        tqdm = None

    errors = []
    with requests.Session() as session:
        session.headers["User-Agent"] = USER_AGENT
        for u in urls:
            for attempt in range(1, retries + 1):
                try:
                    getLogger().info("Downloading {} ...".format(u))
                    with session.get(u, stream=True, timeout=timeout, allow_redirects=True) as r:
                        r.raise_for_status()
                        total = int(r.headers.get("content-length", 0) or 0)
                        bar = None
                        if tqdm is not None:
                            bar = tqdm(
                                total=total or None,
                                unit="B",
                                unit_scale=True,
                                unit_divisor=1024,
                                desc=os.path.basename(local_filename),
                                disable=None,
                                leave=False,
                            )
                        n_bytes = 0
                        try:
                            with open(tmp_filename, "wb") as f:
                                for chunk in r.iter_content(chunk_size=chunk_size):
                                    if chunk:  # filter out keep-alive chunks
                                        f.write(chunk)
                                        n_bytes += len(chunk)
                                        if bar is not None:
                                            bar.update(len(chunk))
                        finally:
                            if bar is not None:
                                bar.close()
                    if total and n_bytes != total and "content-encoding" not in r.headers:
                        raise IOError("incomplete download ({} of {} bytes)".format(n_bytes, total))
                    if validate is not None:
                        validate(tmp_filename)
                    os.replace(tmp_filename, local_filename)
                    return local_filename
                except Exception as e:
                    errors.append("{} (attempt {}): {}".format(u, attempt, e))
                    getLogger().warning("Download of {} failed: {}".format(u, e))
                    if os.path.exists(tmp_filename):
                        os.remove(tmp_filename)
                    # Do not retry the same URL if it is simply missing or its
                    # content is wrong; move straight on to the next mirror.
                    status = getattr(getattr(e, "response", None), "status_code", None)
                    if status in (401, 403, 404, 410) or isinstance(e, (ValueError, tarfile.TarError)):
                        break
                    if attempt < retries:
                        time.sleep(min(2 ** attempt, 30))

    raise DownloadError(
        "Could not download {} from any of {} candidate URL(s):\n  {}".format(
            os.path.basename(local_filename), len(urls), "\n  ".join(errors)
        )
    )


def check_tarball(filename, pattern=None, max_files=5):
    """Validate that ``filename`` is a readable tar archive.

    If ``pattern`` (a compiled regex or string) is given, one of the first
    ``max_files`` regular files in the archive must match it; otherwise
    :class:`ValueError` is raised.  Useful to reject HTML error pages or
    unrelated data products served under the expected URL, without having to
    decompress a (possibly multi-GB) archive in full.
    """
    import re

    regex = re.compile(pattern) if isinstance(pattern, str) else pattern
    n_files = 0
    with tarfile.open(filename) as tar:
        for member in tar:
            if not member.isfile():
                continue
            n_files += 1
            if regex is None or regex.search(member.name):
                return True
            if n_files >= max_files:
                break
    if n_files == 0:
        raise ValueError("{} contains no files.".format(filename))
    raise ValueError("{} does not contain the expected files (pattern {!r}).".format(filename, regex.pattern))


def extract_tarball(filename, directory):
    """Safely extract a tar archive into ``directory``.

    Uses the ``data`` extraction filter (PEP 706) where available, which
    refuses absolute paths, ``..`` components and other unsafe members.
    """
    os.makedirs(directory, exist_ok=True)
    with tarfile.open(filename) as tar:
        if hasattr(tarfile, "data_filter"):
            tar.extractall(directory, filter="data")
        else:  # pragma: no cover - Python < 3.10.12 / 3.11.4
            root = os.path.realpath(directory)
            for member in tar.getmembers():
                target = os.path.realpath(os.path.join(directory, member.name))
                if os.path.commonpath([root, target]) != root:
                    raise tarfile.TarError("Unsafe path in archive: {}".format(member.name))
            tar.extractall(directory)


def grouped_gradient(df, y, x_col, levels=(0, 1), desc=None):
    """``np.gradient(y, x)`` computed separately within each group of a MultiIndexed frame.

    Groups are defined by the given index ``levels`` (e.g. each (age, [Fe/H])
    isochrone or (mass, [Fe/H]) track).  Rows are used in frame order, which
    for the model grids is ascending EEP.  Groups with fewer than two rows get
    NaN.  Returns a float array aligned with ``df``.

    This replaces the old pattern of assigning into ``df.loc[a, b]`` slices,
    which silently does nothing under pandas copy-on-write (pandas >= 3).
    """
    y = np.asarray(y, dtype=float)
    x = df[x_col].to_numpy(dtype=float)
    out = np.full(len(df), np.nan)
    groups = df.groupby(level=list(levels), sort=False).indices
    items = groups.values()
    if desc is not None:
        try:
            from tqdm.auto import tqdm

            items = tqdm(items, total=len(groups), desc=desc, disable=None)
        except ImportError:  # pragma: no cover
            pass
    for idx in items:
        if len(idx) < 2:
            continue
        out[idx] = np.gradient(y[idx], x[idx])
    return out


def addmags(*mags):
    """
    mags is either list of magnitudes or list of (mag, err) pairs
    """
    tot = 0
    uncs = []
    for mag in mags:
        try:
            tot += 10 ** (-0.4 * mag)
        except TypeError:
            m, dm = mag
            f = 10 ** (-0.4 * m)
            tot += f
            unc = f * (1 - 10 ** (-0.4 * dm))
            uncs.append(unc)

    totmag = -2.5 * np.log10(tot)
    if len(uncs) > 0:
        f_unc = np.sqrt(np.array([u ** 2 for u in uncs]).sum())
        return totmag, -2.5 * np.log10(1 - f_unc / tot)
    else:
        return totmag


@nb.jit(nopython=True)
def fast_addmags(mags):
    """
    mags is list of magnitudes
    """
    tot = 0
    for mag in mags:
        tot += 10 ** (-0.4 * mag)
    return -2.5 * log10(tot)


def distance(pos0, pos1):
    """distance between two positions defined by (separation, PA)
    """
    r0, pa0 = pos0
    # getLogger().debug('r0={}, pa0={} (from {})'.format(r0, pa0, self))
    ra0 = r0 * np.sin(pa0 * np.pi / 180)
    dec0 = r0 * np.cos(pa0 * np.pi / 180)

    r1, pa1 = pos1
    # getLogger().debug('r1={}, pa1={} (from {})'.format(r0, pa0, other))
    ra1 = r1 * np.sin(pa1 * np.pi / 180)
    dec1 = r1 * np.cos(pa1 * np.pi / 180)

    dra = ra1 - ra0
    ddec = dec1 - dec0
    return np.sqrt(dra ** 2 + ddec ** 2)


@nb.jit(nopython=True)
def trapz(y, x):
    n = len(y)
    tot = 0

    for i in range(n - 1):
        i2 = i + 1
        tot = tot + 0.5 * (y[i] + y[i2]) * (x[i2] - x[i])

    return tot


@nb.jit(nopython=True)
def polyval(p, x):
    N = len(p)
    result = 0
    for i in range(N):
        result += p[i] * x ** (N - 1 - i)
    return result
