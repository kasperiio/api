"""Free EEX EUA primary-auction prices (EUR/tCO2).

This is an auction-price proxy, not a front-December futures settlement.
Values become usable at midnight UTC after the auction day, conservatively
avoiding same-day publication ambiguity. Only the EUA contract T3PA is used.
"""

from io import BytesIO
import re
import xml.etree.ElementTree as ET
import zipfile

import pandas as pd
import requests

from app.forecasting import series as S

NS = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
BASE_URL = "https://public.eex-group.com/eex/eua-auction-report/"


def parse_eua_workbook(payload):
    # Read the small, fixed-format official XLSX using the standard library;
    # no spreadsheet engine is needed for its stored cell values.
    with zipfile.ZipFile(BytesIO(payload)) as arc:
        shared = ET.fromstring(arc.read("xl/sharedStrings.xml"))
        strings = ["".join(el.itertext()) for el in shared]
        sheet = ET.fromstring(arc.read("xl/worksheets/sheet1.xml"))
    rows = []
    headers = None
    for row in sheet.findall(".//s:row", NS):
        cells = {}
        for cell in row.findall("s:c", NS):
            column = re.sub(r"\d", "", cell.attrib["r"])
            value = cell.findtext("s:v", default="", namespaces=NS)
            cells[column] = strings[int(value)] if cell.attrib.get("t") == "s" else value
        if cells.get("B") == "Date" and cells.get("E") == "Contract":
            headers = cells
            if not cells.get("G", "").startswith("Auction Price"):
                raise ValueError("EEX auction price column changed")
        if cells.get("E") != "T3PA" or cells.get("F", "").lower() != "successful":
            continue
        if headers is None:
            raise ValueError("EEX workbook header missing")
        date = pd.Timestamp("1899-12-30", tz="UTC") + pd.Timedelta(days=float(cells["B"]))
        rows.append({"timestamp": date.normalize() + pd.Timedelta(days=1),
                     "series": S.EUA_AUCTION_EUR_T, "value": float(cells["G"])})
    if headers is None:
        raise ValueError("Unrecognised EEX auction workbook")
    frame = pd.DataFrame(rows, columns=["timestamp", "series", "value"])
    if frame.empty:
        return frame
    return frame.groupby(["timestamp", "series"], as_index=False).value.mean()


class CommoditiesClient:
    def __init__(self, session=None):
        self.session = session or requests.Session()

    def fetch_drivers(self, start, end):
        start, end = pd.Timestamp(start).tz_convert("UTC"), pd.Timestamp(end).tz_convert("UTC")
        end = min(end, pd.Timestamp.now(tz="UTC"))
        if end <= start:
            return pd.DataFrame(columns=["timestamp", "series", "value"])
        frames = []
        for year in range(max(2020, (start - pd.Timedelta(days=1)).year), end.year + 1):
            url = BASE_URL + f"emission-spot-primary-market-auction-report-{year}-data.xlsx"
            response = self.session.get(url, timeout=(10, 60))
            response.raise_for_status()
            frames.append(parse_eua_workbook(response.content))
        if not frames:
            return pd.DataFrame(columns=["timestamp", "series", "value"])
        frame = pd.concat(frames, ignore_index=True)
        return frame[(frame.timestamp >= start) & (frame.timestamp < end)].reset_index(drop=True)
