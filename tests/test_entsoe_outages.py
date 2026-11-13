from io import BytesIO
import zipfile

import pandas as pd

from app.forecasting.sources.entsoe_outages import parse_outage_document, _documents, fetch_outages


def document(status="A05"):
    return f'''<Unavailability_MarketDocument xmlns="urn:test">
    <mRID>ol3</mRID><revisionNumber>2</revisionNumber>
    <createdDateTime>2025-03-01T10:00Z</createdDateTime><docStatus><value>{status}</value></docStatus>
    <TimeSeries><businessType>A53</businessType>
    <production_RegisteredResource.mRID>unit</production_RegisteredResource.mRID>
    <production_RegisteredResource.pSRType.psrType>B14</production_RegisteredResource.pSRType.psrType>
    <production_RegisteredResource.pSRType.powerSystemResources.nominalP>1600</production_RegisteredResource.pSRType.powerSystemResources.nominalP>
    <Available_Period><timeInterval><start>2025-03-02T00:00Z</start><end>2025-03-02T04:00Z</end></timeInterval>
    <resolution>PT60M</resolution><Point><position>1</position><quantity>0</quantity></Point>
    <Point><position>3</position><quantity>800</quantity></Point></Available_Period></TimeSeries>
    </Unavailability_MarketDocument>'''.encode()


def test_parse_generation_intervals_and_publication():
    frame = parse_outage_document(document(), "generation")
    assert list(frame.unavailable_mw) == [1600, 800]
    assert frame.iloc[0].end == pd.Timestamp("2025-03-02T02:00Z")
    assert frame.iloc[0].published_at == pd.Timestamp("2025-03-01T10:00Z")
    assert set(frame.revision) == {2}
    assert set(parse_outage_document(document("A09"), "generation").status) == {"cancelled"}


def test_zip_and_plain_xml():
    xml = document()
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w") as arc:
        arc.writestr("outage.xml", xml)
        arc.writestr("readme.txt", "ignored")
    assert _documents(buf.getvalue()) == [xml]
    assert _documents(xml) == [xml]


def test_cancellation_without_timeseries_is_retained():
    xml = b'''<Unavailability_MarketDocument><mRID>ol3</mRID><revisionNumber>3</revisionNumber>
    <createdDateTime>2025-03-01T11:00Z</createdDateTime><docStatus><value>A09</value></docStatus>
    </Unavailability_MarketDocument>'''
    frame = parse_outage_document(xml, 'generation')
    assert len(frame) == 1
    assert frame.iloc[0].status == 'cancelled'
    assert frame.iloc[0].revision == 3


def test_fetch_paginates_zip_documents_and_requests_both_border_directions():
    from app.forecasting.sources.entsoe_data import EntsoeDataClient, EntsoeNoData
    client = EntsoeDataClient(api_key="test", max_workers=1)
    calls = []

    def get(params, binary=False):
        calls.append(params)
        if params["documentType"] == "A78":
            raise EntsoeNoData("none")
        count = 200 if params["offset"] == "0" else 1
        buf = BytesIO()
        with zipfile.ZipFile(buf, "w") as arc:
            for i in range(count):
                xml = document().replace(b"<mRID>ol3</mRID>",
                                         f'<mRID>{params["offset"]}-{i}</mRID>'.encode())
                arc.writestr(f"{i}.xml", xml)
        return buf.getvalue()

    client._get = get
    frame = fetch_outages(client, pd.Timestamp("2025-03-01T00:00Z"), pd.Timestamp("2025-03-03T00:00Z"))
    assert len(frame) == 402
    assert [c["offset"] for c in calls if c["documentType"] == "A80" and not c.get("docStatus")] == ["0", "200"]
    assert any(c.get("docStatus") == "A13" for c in calls)
    borders = {(c["in_Domain"], c["out_Domain"]) for c in calls if c["documentType"] == "A78"}
    assert len(borders) == 6
