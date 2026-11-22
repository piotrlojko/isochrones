import pytest

from isochrones.query import Query, TwoMASS, WISE, Tycho2, Gaia


@pytest.mark.network
def test_queries(
    ra=45.03433035439128,
    dec=0.23539164875137225,
    pmra=43.75231341609215,
    pmdec=-7.6419899883511482,
    epoch=2015.0,
):
    """Testing with first entry from Gaia DR1 TGAS table (live VizieR, any mirror)
    """
    q = Query(ra, dec, pmra=pmra, pmdec=pmdec, epoch=epoch)

    tm = TwoMASS(q)
    w = WISE(q)
    tyc = Tycho2(q)

    assert str(tm.get_id()) == "03000819+0014074"
    assert str(w.get_id()) == "J030008.22+001407.4"
    assert tyc.get_id() == "55-256-1"

    phot = tm.get_photometry()
    assert set(phot) == {"J", "H", "K"}


@pytest.mark.network
def test_gaia_query():
    q = Query(45.03433035439128, 0.23539164875137225, pmra=43.75, pmdec=-7.64, epoch=2015.0)
    gaia = Gaia(q)
    table = gaia.table
    assert len(table) >= 1
    assert table["Source"][0] == 7632157690368
    # Quality flags are computed (this TGAS star fails the Nper > 8 cut of
    # Lindegren et al. 2018, so it is correctly flagged as not good).
    assert table["is_good"].dtype == bool
