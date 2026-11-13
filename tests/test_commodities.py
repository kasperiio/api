from io import BytesIO
import zipfile

import pandas as pd

from app.forecasting import features as F, series as S, store
from app.forecasting.sources.commodities import parse_eua_workbook
from tests.test_store import db


def workbook():
    ns = 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'
    strings = ['Date', 'Contract', 'Auction Price EUR/tCO2', 'T3PA', 'successful', 'T3AA']
    shared = '<sst xmlns="' + ns + '">' + ''.join('<si><t>' + s + '</t></si>' for s in strings) + '</sst>'
    sheet = f'''<worksheet xmlns="{ns}"><sheetData>
      <row r="6"><c r="B6" t="s"><v>0</v></c><c r="E6" t="s"><v>1</v></c><c r="G6" t="s"><v>2</v></c></row>
      <row r="7"><c r="B7"><v>45658</v></c><c r="E7" t="s"><v>3</v></c><c r="F7" t="s"><v>4</v></c><c r="G7"><v>75</v></c></row>
      <row r="8"><c r="B8"><v>45658</v></c><c r="E8" t="s"><v>5</v></c><c r="F8" t="s"><v>4</v></c><c r="G8"><v>999</v></c></row>
      </sheetData></worksheet>'''
    buf = BytesIO()
    with zipfile.ZipFile(buf, 'w') as arc:
        arc.writestr('xl/sharedStrings.xml', shared)
        arc.writestr('xl/worksheets/sheet1.xml', sheet)
    return buf.getvalue()


def test_only_eua_auction_and_available_next_day():
    frame = parse_eua_workbook(workbook())
    assert list(frame.value) == [75.0]
    assert frame.iloc[0].timestamp == pd.Timestamp('2025-01-02', tz='UTC')
    assert frame.iloc[0].series == S.EUA_AUCTION_EUR_T


def test_carbon_never_backfills_before_publication_and_expires(db):
    frame = parse_eua_workbook(workbook())
    store.upsert_driver_data(db, frame, 'eex')
    values = F._load_drivers(db, pd.Timestamp('2025-01-01', tz='UTC'),
                             pd.Timestamp('2025-01-12', tz='UTC'))[S.EUA_AUCTION_EUR_T]
    assert pd.isna(values.loc['2025-01-01T12:00Z'])
    assert values.loc['2025-01-02T00:00Z'] == 75.0
    assert pd.isna(values.loc['2025-01-10T00:00Z'])
