import os
import re
import glob
import warnings

import pandas as pd

from .config import ISOCHRONES
from .grid import Grid
from .logger import getLogger


class BolometricCorrectionGrid(Grid):
    """Bolometric corrections in different bands, as a function of stuff

    Stores bolometric corrections computed on a grid of stellar atmospheric
    parameters (Teff, logg, [Fe/H]), Av, and Rv.

    Specific implementations of this grid should subclass this
    (e.g., `MISTBolometricCorrectionGrid`).

    Parameters
    ----------
    bands : list(str)
        List of band names, each parsed with `get_band` method.
        Tables are downloaded when requested.
    """

    index_cols = ("Teff", "logg", "[Fe/H]", "Av", "Rv")
    name = None
    is_full = True

    def __init__(self, bands=None):

        self.bands = bands if bands is not None else list(self.default_bands)

        self._band_map = None
        self._phot_systems = None

        self._df = None
        self._interp = None

    def get_band(self, *args, **kwargs):
        return NotImplementedError

    def _make_band_map(self):
        phot_systems = set()
        band_map = {}
        for b in self.bands:
            phot, band = self.get_band(b)
            phot_systems.add(phot)
            band_map[b] = band
        self._band_map = band_map
        self._phot_systems = phot_systems

    @property
    def band_map(self):
        if self._band_map is None:
            self._make_band_map()
        return self._band_map

    @property
    def phot_systems(self):
        if self._phot_systems is None:
            self._make_band_map()
        return self._phot_systems

    @property
    def datadir(self):
        return os.path.join(ISOCHRONES, "BC", self.name)

    def get_filename(self, phot, feh):
        rootdir = self.datadir
        sign_str = "m" if feh < 0 else "p"
        filename = "feh{0}{1:03.0f}.{2}".format(sign_str, abs(feh) * 100, phot)
        return os.path.join(rootdir, filename)

    @staticmethod
    def get_table_pattern(phot):
        """Regex matching the names of the individual BC tables of a photometric system.

        e.g. ``fehm050.UBVRIplus`` or ``fehp025.WISE``.
        """
        return re.compile(r"(^|/)feh[mp]\d{{3}}\.{}$".format(re.escape(phot)))

    def get_table_filenames(self, phot):
        """BC table files for ``phot`` present in the data directory.

        Only files following the ``feh[mp]NNN.<phot>`` naming convention of the
        BC tables are returned, so unrelated files that happen to share the
        extension (e.g. MIST v2 isochrones ``*_full.iso.UBVRIplus``) are ignored.
        """
        pattern = self.get_table_pattern(phot)
        return sorted(f for f in glob.glob(os.path.join(self.datadir, "*.{}".format(phot))) if pattern.search(f))

    def parse_table(self, filename):
        """Reads text table into dataframe
        """
        names = None
        with open(filename) as fin:
            for i, line in enumerate(fin):
                if not line.startswith("#"):
                    break
                cols = line[1:].split()
                # The column-name line is the one that starts with the index columns
                if cols[: len(self.index_cols)] == list(self.index_cols):
                    names = cols
                    break
        if names is None:
            raise ValueError("{} does not look like a bolometric correction table.".format(filename))
        return pd.read_csv(filename, names=names, sep=r"\s+", comment="#", index_col=list(self.index_cols))

    def get_table(self, phot, feh):
        return self.parse_table(self.get_filename(phot, feh))

    def get_hdf_filename(self, phot):
        return os.path.join(self.datadir, "{}.h5".format(phot))

    def get_tarball_url(self, phot):
        return self.get_tarball_urls(phot)[0]

    def get_tarball_urls(self, phot):
        raise NotImplementedError

    def get_tarball_member_pattern(self, phot):
        return self.get_table_pattern(phot)

    def get_tarball_file(self, phot):
        return os.path.join(self.datadir, "{}.txz".format(phot))

    def get_df(self):
        dfs = []
        for phot in sorted(self.phot_systems):
            hdf_filename = self.get_hdf_filename(phot=phot)
            if not os.path.exists(hdf_filename):
                filenames = self.get_table_filenames(phot)
                if not filenames:
                    self.extract_tarball(phot=phot)
                    filenames = self.get_table_filenames(phot)
                if not filenames:
                    raise FileNotFoundError(
                        "No {} bolometric correction tables found in {}.".format(phot, self.datadir)
                    )
                getLogger().info("Parsing {} {} bolometric correction tables...".format(len(filenames), phot))
                df = pd.concat([self.parse_table(f) for f in filenames]).sort_index()
                with warnings.catch_warnings():
                    # "[Fe/H]" is not a valid PyTables natural name; harmless.
                    warnings.filterwarnings("ignore", message="object name is not a valid Python identifier")
                    df.to_hdf(hdf_filename, key="df")
            df = pd.read_hdf(hdf_filename, key="df")
            dfs.append(df)

        df_all = pd.concat(dfs, axis=1) if dfs else pd.DataFrame()

        df_all = df_all.rename(columns={v: k for k, v in self.band_map.items()})
        df_all = df_all[[c for c in df_all.columns if c in self.bands]]

        return df_all
