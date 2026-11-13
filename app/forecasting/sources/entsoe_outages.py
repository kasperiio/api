"""ENTSO-E outage documents, preserving publication time and revisions.

Generation uses A80; transmission uses A78 in both border directions. The API
returns zipped XML documents and at most 200 documents per page. Transmission
reports available capacity, often without a nominal capacity: unavailable MW
is left missing in that case rather than inventing a link rating.
"""

from io import BytesIO
import zipfile
import xml.etree.ElementTree as ET

import pandas as pd

from app.forecasting.store import OUTAGE_COLUMNS
from app.forecasting.sources.entsoe_data import EntsoeNoData, EntsoeError, ZONE_EIC


def parse_outage_document(xml, kind, border=""):
    root = ET.fromstring(xml)
    for el in root.iter():
        el.tag = el.tag.split("}")[-1]

    def txt(el, path, default=""):
        return el.findtext(path, default=default)

    def number(text):
        return float(text) if text else None

    mrid = txt(root, "mRID")
    published = pd.Timestamp(txt(root, "createdDateTime"))
    if not mrid or published.tzinfo is None:
        raise ValueError("Outage document lacks identity or UTC publication time")
    revision = int(txt(root, "revisionNumber", "1"))
    status = "cancelled" if txt(root, "docStatus/value") in ("A09", "A13") else "active"
    rows = []
    for ts in root.findall("TimeSeries"):
        nominal = number(txt(ts, "production_RegisteredResource.pSRType.powerSystemResources.nominalP"))
        resource = txt(ts, "production_RegisteredResource.mRID")
        psr = txt(ts, "production_RegisteredResource.pSRType.psrType")
        for period in ts.findall("Available_Period"):
            start = pd.Timestamp(txt(period, "timeInterval/start"))
            end = pd.Timestamp(txt(period, "timeInterval/end"))
            resolution = pd.Timedelta(txt(period, "resolution"))
            points = sorted(period.findall("Point"), key=lambda p: int(txt(p, "position")))
            for i, point in enumerate(points):
                lo = start + (int(txt(point, "position")) - 1) * resolution
                hi = (start + (int(txt(points[i + 1], "position")) - 1) * resolution
                      if i + 1 < len(points) else end)
                available = number(txt(point, "quantity"))
                unavailable = max(0.0, nominal - available) if nominal is not None and available is not None else None
                rows.append(dict(mrid=mrid, revision=revision, published_at=published,
                                 kind=kind, business_type="planned" if txt(ts, "businessType") == "A53" else "forced",
                                 status=status, resource=resource, border=border, psr_type=psr,
                                 nominal_mw=nominal, start=lo, end=hi,
                                 available_mw=available, unavailable_mw=unavailable))
    if not rows and status == "cancelled":
        # Cancellation notices may omit the original time series. Keep a
        # tombstone so earlier revisions cannot reappear in as-of queries.
        rows.append(dict(mrid=mrid, revision=revision, published_at=published,
                         kind=kind, business_type="planned", status=status,
                         resource="", border=border, psr_type="", nominal_mw=None,
                         start=pd.Timestamp("1970-01-01", tz="UTC"),
                         end=pd.Timestamp("1970-01-01T01:00Z"),
                         available_mw=None, unavailable_mw=None))
    return pd.DataFrame(rows, columns=OUTAGE_COLUMNS)


def _documents(payload):
    if payload[:2] == b"PK":
        with zipfile.ZipFile(BytesIO(payload)) as arc:
            return [arc.read(f) for f in arc.namelist() if f.lower().endswith(".xml")]
    return [payload]


def fetch_outages(client, start, end):
    start, end = pd.Timestamp(start).tz_convert("UTC"), pd.Timestamp(end).tz_convert("UTC")
    queries = [("generation", "", {"documentType": "A80", "biddingZone_Domain": ZONE_EIC["FI"]})]
    for zone in ("EE", "SE1", "SE3"):
        for a, b in (("FI", zone), (zone, "FI")):
            queries.append(("transmission", f"{a}-{b}",
                            {"documentType": "A78", "out_Domain": ZONE_EIC[a], "in_Domain": ZONE_EIC[b]}))
    # ENTSO-E's default query includes active/cancelled notices. Withdrawn
    # notices need an explicit request, otherwise an older stored outage can
    # remain active after its document has been withdrawn.
    queries += [(kind, border, dict(query, docStatus="A13"))
                for kind, border, query in list(queries)]
    windows = []
    lo = start
    while lo < end:
        hi = min(lo + pd.Timedelta(days=92), end)
        windows.append((lo, hi))
        lo = hi

    def fetch(task):
        lo, hi, kind, border, query = task
        frames = []
        for offset in range(0, 4801, 200):
            params = dict(query, periodStart=client._fmt(lo), periodEnd=client._fmt(hi), offset=str(offset))
            try:
                docs = _documents(client._get(params, binary=True))
            except EntsoeNoData:
                break
            except EntsoeError as exc:
                if hi - lo <= pd.Timedelta(days=31) or not any(
                    f"HTTP {code}" in str(exc) for code in (500, 502, 503, 504, 599)
                ):
                    raise
                # Some broad historical queries time out upstream. Retry in
                # monthly windows while preserving all document revisions.
                pieces = []
                cur = lo
                while cur < hi:
                    nxt = min(cur + pd.Timedelta(days=31), hi)
                    pieces.extend(fetch((cur, nxt, kind, border, query)))
                    cur = nxt
                return pieces
            frames.extend(parse_outage_document(doc, kind, border) for doc in docs)
            if len(docs) < 200:
                break
        else:
            raise EntsoeError("Outage pagination limit reached; use smaller date windows")
        return frames

    tasks = [(lo, hi, kind, border, query) for lo, hi in windows for kind, border, query in queries]
    frames = [f for result in client._map(fetch, tasks) for f in result if not f.empty]
    if not frames:
        return pd.DataFrame(columns=OUTAGE_COLUMNS)
    return pd.concat(frames, ignore_index=True).drop_duplicates(["mrid", "revision", "start"])
