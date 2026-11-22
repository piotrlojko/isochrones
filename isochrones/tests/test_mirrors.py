"""Offline tests for download/mirror fallbacks and pandas>=3 helpers.

None of these touch the network: a throw-away local HTTP server stands in for
remote mirrors, and VizieR responses are mocked.
"""
import io
import os
import tarfile
import threading
import functools
from http.server import HTTPServer, SimpleHTTPRequestHandler

import numpy as np
import pandas as pd
import pytest

from isochrones import utils
from isochrones.utils import download_file, check_tarball, DownloadError, grouped_gradient
from isochrones.interp import DFInterpolator


class _QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


@pytest.fixture
def http_root(tmp_path):
    """Serve ``tmp_path / "www"`` on a random localhost port."""
    www = tmp_path / "www"
    www.mkdir()
    handler = functools.partial(_QuietHandler, directory=str(www))
    server = HTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield www, "http://127.0.0.1:{}".format(server.server_address[1])
    finally:
        server.shutdown()
        server.server_close()


def _make_tarball(path, members):
    with tarfile.open(path, "w:xz") as tar:
        for name, content in members.items():
            data = content.encode()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))


def test_download_falls_back_to_next_mirror(http_root, tmp_path):
    www, base = http_root
    (www / "mirror2").mkdir()
    (www / "mirror2" / "file.txt").write_text("hello")

    dest = tmp_path / "out" / "file.txt"
    urls = [base + "/mirror1/file.txt", base + "/mirror2/file.txt"]
    download_file(urls, str(dest), retries=1, timeout=5)

    assert dest.read_text() == "hello"
    assert not os.path.exists(str(dest) + ".part")


def test_download_rejects_wrong_content(http_root, tmp_path):
    www, base = http_root
    (www / "a").mkdir()
    (www / "b").mkdir()
    # Mirror "a" serves an HTML error page under the tarball name ...
    (www / "a" / "WISE.txz").write_text("<html>Service unavailable</html>")
    # ... mirror "b" serves a tarball with unrelated content ...
    _make_tarball(str(www / "b" / "WISE.txz"), {"feh_m025_afe_p0_vvcrit0.4_full.iso.WISE": "x"})
    # ... and mirror "c" the real thing.
    (www / "c").mkdir()
    _make_tarball(str(www / "c" / "WISE.txz"), {"fehm025.WISE": "x", "fehp000.WISE": "y"})

    from isochrones.bc import BolometricCorrectionGrid

    pattern = BolometricCorrectionGrid.get_table_pattern("WISE")
    dest = tmp_path / "WISE.txz"
    download_file(
        [base + "/a/WISE.txz", base + "/b/WISE.txz", base + "/c/WISE.txz"],
        str(dest),
        retries=1,
        timeout=5,
        validate=lambda f: check_tarball(f, pattern),
    )
    with tarfile.open(dest) as tar:
        assert sorted(tar.getnames()) == ["fehm025.WISE", "fehp000.WISE"]


def test_download_all_mirrors_fail(http_root, tmp_path):
    _, base = http_root
    with pytest.raises(DownloadError):
        download_file([base + "/nope1.txz", base + "/nope2.txz"], str(tmp_path / "x.txz"), retries=1, timeout=5)
    assert not (tmp_path / "x.txz").exists()


def test_bc_table_filenames_ignore_other_products(tmp_path, monkeypatch):
    from isochrones.mist.bc import MISTBolometricCorrectionGrid

    monkeypatch.setattr(MISTBolometricCorrectionGrid, "datadir", str(tmp_path))
    for name in [
        "fehm050.UBVRIplus",
        "fehp025.UBVRIplus",
        # MIST v2.5 synthetic-photometry isochrones share the extension:
        "feh_m050_afe_p0_vvcrit0.4_full.iso.UBVRIplus",
        "UBVRIplus.txz",
    ]:
        (tmp_path / name).write_text("")

    bc = MISTBolometricCorrectionGrid(["J"])
    names = [os.path.basename(f) for f in bc.get_table_filenames("UBVRIplus")]
    assert names == ["fehm050.UBVRIplus", "fehp025.UBVRIplus"]


def test_bc_parse_table_rejects_isochrone_files(tmp_path):
    from isochrones.mist.bc import MISTBolometricCorrectionGrid

    iso = tmp_path / "feh_m025_afe_p0_vvcrit0.4_full.iso.WISE"
    iso.write_text("# MIST version number  = 2.5\n# EEP log10_isochrone_age_yr initial_mass\n 1 5.0 1.0\n")
    with pytest.raises(ValueError):
        MISTBolometricCorrectionGrid(["W1"]).parse_table(str(iso))

    bc = tmp_path / "fehp000.WISE"
    bc.write_text(
        "# WISE (Vega)\n#  filters spectra  num Av  num Rv\n#        4    1820      13       1\n#\n"
        "#      1    2     3     4     5   6   7   8   9\n"
        "#  Teff logg [Fe/H]   Av    Rv   WISE_W1   WISE_W2   WISE_W3   WISE_W4\n"
        "   2500. -4.0  0.00  0.00  3.10   3.37  3.55  3.87  4.08\n"
    )
    df = MISTBolometricCorrectionGrid(["W1"]).parse_table(str(bc))
    assert list(df.index.names) == ["Teff", "logg", "[Fe/H]", "Av", "Rv"]
    assert df["WISE_W1"].iloc[0] == pytest.approx(3.37)


def test_mist_urls_cover_all_mirrors():
    from isochrones import config
    from isochrones.mist.bc import MISTBolometricCorrectionGrid
    from isochrones.mist.models import MISTIsochroneGrid, MISTEvolutionTrackGrid

    bc_urls = MISTBolometricCorrectionGrid(["J"]).get_tarball_urls("UBVRIplus")
    assert bc_urls[0] == config.MIST_MIRRORS[0].rstrip("/") + "/BC_tables/v1/UBVRIplus.txz"
    assert len(bc_urls) == 2 * len(config.MIST_MIRRORS)

    iso_urls = MISTIsochroneGrid().get_tarball_urls()
    assert [u.endswith("data/tarballs_v1.2/MIST_v1.2_vvcrit0.4_full_isos.txz") for u in iso_urls] == [True] * len(
        config.MIST_MIRRORS
    )

    track_urls = MISTEvolutionTrackGrid().get_tarball_urls(feh=-0.25)
    assert track_urls[0].endswith("data/tarballs_v1.2/MIST_v1.2_feh_m0.25_afe_p0.0_vvcrit0.4_EEPS.txz")


def test_env_list(monkeypatch):
    from isochrones.config import _env_list

    monkeypatch.setenv("ISOCHRONES_TEST_LIST", " https://a.example ,http://b.example/,, ")
    assert _env_list("ISOCHRONES_TEST_LIST", ("x",)) == ("https://a.example", "http://b.example/")
    monkeypatch.delenv("ISOCHRONES_TEST_LIST")
    assert _env_list("ISOCHRONES_TEST_LIST", ("x",)) == ("x",)


class _FakeResponse:
    def __init__(self, text, status=200):
        self.text = text
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests

            raise requests.HTTPError("{} error".format(self.status_code), response=self)


def test_vizier_mirror_fallback(monkeypatch):
    import requests
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    from isochrones.query import vizier

    calls = []

    def fake_async(self, coords, radius=None, catalog=None, cache=True, **kw):
        calls.append(self.VIZIER_SERVER)
        if self.VIZIER_SERVER == "down.example":
            raise requests.ConnectionError("connection refused")
        if self.VIZIER_SERVER == "http-404.example":
            return _FakeResponse("not found", status=404)
        if self.VIZIER_SERVER == "partial.example":
            return _FakeResponse('<INFO ID="Error" name="Error" value="Table or Catalog not found: {}"/>'.format(catalog))
        return _FakeResponse('<INFO name="QUERY_STATUS" value="OK"/>ok')

    monkeypatch.setattr(vizier.MirroredVizier, "query_region_async", fake_async)
    monkeypatch.setattr(vizier.MirroredVizier, "_parse_result", lambda self, response, **kw: ["table"])
    monkeypatch.setattr(vizier, "_dead_mirrors", {})

    mirrors = ["down.example", "http-404.example", "partial.example", "http://good.example"]
    result, server = vizier.query_region_with_mirrors(
        SkyCoord(10, 10, unit="deg"), 5 * u.arcsec, "2mass", mirrors=mirrors, timeout=1
    )
    assert result == ["table"]
    assert server == "http://good.example"
    assert calls == mirrors

    # Unreachable mirrors are demoted for subsequent queries
    monkeypatch.setattr(vizier, "VIZIER_MIRRORS", tuple(mirrors))
    monkeypatch.setenv("ISOCHRONES_VIZIER_MIRRORS", ",".join(mirrors))
    order = vizier.get_vizier_mirrors()
    assert order[-2:] == ["down.example", "http-404.example"]


def test_vizier_all_mirrors_fail(monkeypatch):
    import requests
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    from isochrones.query import vizier

    def fake_async(self, *args, **kw):
        raise requests.ConnectionError("down")

    monkeypatch.setattr(vizier.MirroredVizier, "query_region_async", fake_async)
    monkeypatch.setattr(vizier, "_dead_mirrors", {})
    with pytest.raises(vizier.VizierUnavailableError):
        vizier.query_region_with_mirrors(SkyCoord(10, 10, unit="deg"), 5 * u.arcsec, "2mass", mirrors=["a", "b"])


def test_mirrored_vizier_http_urls():
    from isochrones.query.vizier import MirroredVizier

    assert MirroredVizier(vizier_server="http://vizier.nao.ac.jp")._server_to_url() == (
        "http://vizier.nao.ac.jp/viz-bin/votable"
    )
    assert MirroredVizier(vizier_server="vizier.cfa.harvard.edu")._server_to_url() == (
        "https://vizier.cfa.harvard.edu/viz-bin/votable"
    )


def test_grouped_gradient_matches_per_group():
    idx = pd.MultiIndex.from_product([[1.0, 2.0], [-0.5, 0.0], np.arange(5)], names=["a", "b", "eep"])
    rng = np.random.default_rng(42)
    df = pd.DataFrame({"eep": idx.get_level_values("eep").astype(float), "y": rng.random(len(idx))}, index=idx)

    out = grouped_gradient(df, df["y"], "eep")
    for key, sub in df.groupby(level=[0, 1]):
        expected = np.gradient(sub["y"].to_numpy(), sub["eep"].to_numpy())
        np.testing.assert_allclose(out[df.index.get_locs(list(key))], expected)


def test_dfinterpolator_pads_incomplete_grid():
    idx = pd.MultiIndex.from_product([[0.0, 1.0], [0.0, 1.0, 2.0]], names=["x", "y"])
    df = pd.DataFrame({"v": np.arange(6, dtype=float), "flag": [True, False] * 3}, index=idx)
    df = df.drop((1.0, 2.0))  # missing grid point -> NaN

    interp = DFInterpolator(df)
    assert interp.grid.shape == (2, 3, 2)
    assert np.isnan(interp.grid[1, 2]).all()
    assert interp([0.5, 0.5], ["v"])[0] == pytest.approx(np.mean([0, 1, 3, 4]))
