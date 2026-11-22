isochrones
==========

Provides simple interface for interacting with stellar model grids.

https://isochrones.readthedocs.io

Installation
------------

Requires Python >= 3.10.  The package is tested against current releases of its
dependencies (numpy 2.x, pandas 2.x/3.x, scipy, astropy 7/8, astroquery, numba,
PyTables, emcee 3)::

    pip install -e .              # from a clone of this repository
    pip install -e ".[test,plot]" # + pytest/flaky, matplotlib/holoviews

`MultiNest <https://github.com/JohannesBuchner/MultiNest>`_ is used for fits by
default.  ``pymultinest`` is installed automatically, but the MultiNest library
itself must be built separately and its ``lib/`` directory added to
``LD_LIBRARY_PATH``.  Without it ``isochrones`` still imports (with a warning)
and fits can be run with ``use_emcee=True``.

Model grids are downloaded on first use into ``~/.isochrones`` (override with
the ``ISOCHRONES`` environment variable).  The MIST isochrone grid needs
~1.5 GB; the evolution tracks another ~1.3 GB download.

Data sources and mirrors
------------------------

* **MIST** model grids and bolometric-correction tables are fetched from
  https://mist.science (the former ``waps.cfa.harvard.edu/MIST`` address
  redirects there).  As a last resort, snapshots on the Internet Archive are
  tried.  Downloaded tarballs are validated, so an HTML error page or an
  unrelated data product is never unpacked into the cache; a cached tarball
  that fails validation is renamed to ``*.invalid`` and fetched again.
* **VizieR** queries (``isochrones.query``) fall back through the official
  mirrors when the CDS server in Strasbourg is unavailable: CfA
  (``vizier.cfa.harvard.edu``), IDIA (``vizier.idia.ac.za``), NAOJ
  (``http://vizier.nao.ac.jp``) and China-VO (``http://vizier.china-vo.org``).
  Mirrors that cannot be reached are skipped for the rest of the session, and
  mirrors reporting a server-side error (e.g. a catalogue they do not host) are
  skipped for that query.  ``isochrones.query.vizier_mirror_status()`` reports
  which mirrors currently answer.
* **Extinction** (``isochrones.extinction.get_AV_infinity``) uses the NED
  extinction calculator, falling back to the IRSA dust service.

The lists can be changed with environment variables (comma-separated, tried in
order)::

    export ISOCHRONES_MIST_MIRRORS="https://mist.science,https://my.mirror/MIST"
    export ISOCHRONES_VIZIER_MIRRORS="vizier.cfa.harvard.edu,http://vizier.nao.ac.jp"
    export ISOCHRONES_DOWNLOAD_TIMEOUT=120   # seconds
    export ISOCHRONES_DOWNLOAD_RETRIES=3     # attempts per URL

A VizieR server set in astroquery's own configuration (``[vizier] server`` in
``astroquery.cfg``) is tried first unless ``ISOCHRONES_VIZIER_MIRRORS`` is set.

Basic Usage
------------

For simplest out-of-the-box usage after installation, make a file called ``star.ini`` that
looks something like this::

    Teff = 5770, 100
    feh = 0.0, 0.15
    logg = 4.5, 0.1
    V = 10.0, 0.05

Any combination of spectroscopic or [supported] photometric properties, as well
as parallax, can go as properties into this file.

Once you've made this file, type ``starfit`` at the command line.  Or if you want to be organized,
you can put ``star.ini`` in a folder called ``mystar`` [or whatever]
and run ``starfit mystar`` from the command line.  The ``starfit`` script
will create an HDF5 file containing the saved ``StarModel``, which you
can load from python using ``StarModel.load_hdf``, as well as triangle
plots illustrating the fit.

Testing
-------

::

    pytest                                  # everything (downloads grids, needs network)
    pytest -m "not slow and not network"    # skip evolution-track builds and live-service tests
    pytest isochrones/tests/test_mirrors.py # offline download/mirror logic only

Attribution
------------
If you use ``isochrones`` in your research, please cite `this ASCL reference <http://adsabs.harvard.edu/cgi-bin/nph-bib_query?bibcode=2015ascl.soft03010M&data_type=BIBTEX&db_key=AST&nocookieset=1>`_.
