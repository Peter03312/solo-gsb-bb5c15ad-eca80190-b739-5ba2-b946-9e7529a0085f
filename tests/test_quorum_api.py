"""End-to-end API behavior for the redundant-sensor quorum endpoint."""


def _event(rev, parent, start, end, value=None, sensor="S1", unit="C",
           withdrawn=False, key=None, batch="B1"):
    return {
        "batch_id": batch,
        "sensor_id": sensor,
        "revision_id": rev,
        "parent_id": parent,
        "idempotency_key": key or f"k-{sensor}-{rev}",
        "start_ns": start,
        "end_ns": end,
        "value": value,
        "unit": unit,
        "withdrawn": withdrawn,
    }


def _post(client, event, batch="B1"):
    return client.post(f"/api/v1/batches/{batch}/events", json=event)


def _vote(client, sensor_ids, batch="B1", **overrides):
    body = {
        "window_start_ns": 0,
        "window_end_ns": 100,
        "sensor_ids": sensor_ids,
        "expected_unit": "C",
        "quorum": 2,
        "max_value_diff": 0.5,
    }
    body.update(overrides)
    return client.post(f"/api/v1/batches/{batch}/redundancy-vote", json=body)


def test_quorum_qualified_full_window(client):
    _post(client, _event("r1", "ROOT", 0, 100, value=10.0, sensor="S1"))
    _post(client, _event("r2", "ROOT", 0, 100, value=10.3, sensor="S2"))
    _post(client, _event("r3", "ROOT", 0, 100, value=10.5, sensor="S3"))
    r = _vote(client, ["S1", "S2", "S3"], quorum=2)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "qualified"
    assert (body["window_start_ns"], body["window_end_ns"]) == (0, 100)
    assert body["quorum"] == 2 and body["max_value_diff"] == 0.5
    assert body["sensor_ids"] == ["S1", "S2", "S3"]
    (span,) = body["intervals"]
    assert span["start_ns"] == 0 and span["end_ns"] == 100
    assert span["duration_ns"] == 100 and span["status"] == "qualified"
    assert [m["sensor_id"] for m in span["members"]] == ["S1", "S2", "S3"]
    assert span["non_votes"] == []
    assert span["largest_compatible_size"] == 3
    assert len(body["input_hash"]) == 64


def test_quorum_strict_majority_threshold_example(client):
    # 3 sensors, q=2; S1 masked by a blocked revision on [60,100),
    # S2=10.2 agrees elsewhere, S3 jumps to 20.0 at t=40.
    _post(client, _event("a", "ROOT", 0, 100, value=10.0, sensor="S1", key="ka"))
    _post(client, _event("a2", "ghost", 60, 100, value=50.0, sensor="S1", key="ka2"))
    _post(client, _event("b", "ROOT", 0, 100, value=10.2, sensor="S2", key="kb"))
    _post(client, _event("c1", "ROOT", 0, 40, value=10.4, sensor="S3", key="kc1"))
    _post(client, _event("c2", "ROOT", 40, 100, value=20.0, sensor="S3", key="kc2"))
    body = _vote(client, ["S1", "S2", "S3"], quorum=2).json()
    assert body["status"] == "qualified"
    assert [(i["start_ns"], i["end_ns"], i["status"]) for i in body["intervals"]] == [
        (0, 40, "qualified"),
        (40, 60, "qualified"),
        (60, 100, "pending_backfill"),
    ]
    first, second, pending = body["intervals"]
    assert [m["sensor_id"] for m in first["members"]] == ["S1", "S2", "S3"]
    assert [m["sensor_id"] for m in second["members"]] == ["S1", "S2"]
    assert second["largest_compatible_size"] == 2
    (nv,) = pending["non_votes"]
    assert nv["sensor_id"] == "S1" and nv["reason"] == "blocked"
    assert nv["missing_ancestors"] == ["ghost"]
    assert pending["missing_ancestors"] == ["ghost"]
    assert pending["blocked_count"] == 1
    assert pending["largest_compatible_size"] == 1  # S2 only (S3=20 disagrees)
    assert pending["members"] == []


def test_quorum_rejected_when_no_trusted_majority_possible(client):
    _post(client, _event("r1", "ROOT", 0, 100, value=1.0, sensor="S1"))
    _post(client, _event("r2", "ROOT", 0, 100, value=5.0, sensor="S2"))
    _post(client, _event("r3", "ROOT", 0, 100, value=9.0, sensor="S3"))
    body = _vote(client, ["S1", "S2", "S3"], quorum=2, max_value_diff=0.0).json()
    assert body["status"] == "rejected"
    (span,) = body["intervals"]
    assert span["status"] == "rejected"
    assert span["members"] == [] and span["missing_ancestors"] == []
    assert span["largest_compatible_size"] == 1 and span["blocked_count"] == 0


def test_quorum_non_vote_reasons_are_reported(client):
    _post(client, _event("root", "ROOT", 0, 80, value=1.0, sensor="S1"))
    _post(client, _event("w", "root", 20, 40, sensor="S1", withdrawn=True, value=None))
    _post(client, _event("f", "w", 40, 60, value=1.0, sensor="S1", unit="K"))
    _post(client, _event("r2", "ROOT", 0, 100, value=1.0, sensor="S2"))
    _post(client, _event("r3", "ROOT", 0, 100, value=1.1, sensor="S3"))
    body = _vote(client, ["S1", "S2", "S3"], quorum=3).json()
    by_span = {i["start_ns"]: i for i in body["intervals"]}
    assert not by_span[0]["non_votes"]
    assert by_span[20]["non_votes"][0]["reason"] == "withdrawn"
    unit_nv = by_span[40]["non_votes"][0]
    assert unit_nv["reason"] == "unit_mismatch"
    assert unit_nv["unit"] == "K" and unit_nv["value"] == 1.0
    assert by_span[80]["non_votes"][0]["reason"] == "uncovered"


def test_quorum_blocked_masks_landed_value(client):
    _post(client, _event("root", "ROOT", 0, 100, value=10.0, sensor="S1"))
    _post(client, _event("o", "ghost", 40, 60, value=99.0, sensor="S1"))
    _post(client, _event("r2", "ROOT", 0, 100, value=99.0, sensor="S2"))
    body = _vote(client, ["S1", "S2"], quorum=2, max_value_diff=0.5).json()
    assert [(i["start_ns"], i["end_ns"], i["status"]) for i in body["intervals"]] == [
        (0, 40, "rejected"),
        (40, 60, "pending_backfill"),
        (60, 100, "rejected"),
    ]
    pending = body["intervals"][1]
    assert pending["missing_ancestors"] == ["ghost"]
    assert pending["non_votes"][0]["reason"] == "blocked"


def test_quorum_unknown_batch_or_sensor_404(client):
    r = _vote(client, ["S1", "S2"])
    assert r.status_code == 404 and "B1" in r.json()["detail"]
    _post(client, _event("r1", "ROOT", 0, 100, value=1.0, sensor="S1"))
    r = _vote(client, ["S1", "S2"])
    assert r.status_code == 404
    assert "S2" in r.json()["detail"] and "B1" in r.json()["detail"]


def test_quorum_invalid_params_422(client):
    _post(client, _event("r1", "ROOT", 0, 100, value=1.0, sensor="S1"))
    _post(client, _event("r2", "ROOT", 0, 100, value=1.1, sensor="S2"))
    _post(client, _event("r3", "ROOT", 0, 100, value=1.2, sensor="S3"))

    # q out of range for 3 sensors (strict majority is 2, total is 3).
    assert _vote(client, ["S1", "S2", "S3"], quorum=1).status_code == 422
    assert _vote(client, ["S1", "S2", "S3"], quorum=4).status_code == 422
    # q for a two-sensor list must be exactly 2.
    assert _vote(client, ["S1", "S2"], quorum=1).status_code == 422
    # Fewer than two sensors / duplicate sensors.
    assert _vote(client, ["S1"]).status_code == 422
    assert _vote(client, ["S1", "S1", "S3"]).status_code == 422
    # Negative or non-finite spread; zero is allowed.
    assert _vote(client, ["S1", "S2", "S3"], max_value_diff=-0.01).status_code == 422
    assert _vote(client, ["S1", "S2", "S3"], max_value_diff=0.0).status_code == 200
    for raw in ("NaN", "Infinity", "-Infinity"):
        payload = (
            '{"window_start_ns":0,"window_end_ns":100,'
            '"sensor_ids":["S1","S2","S3"],"expected_unit":"C","quorum":2,'
            f'"max_value_diff":{raw}}}'
        )
        r = client.post(
            "/api/v1/batches/B1/redundancy-vote",
            content=payload,
            headers={"content-type": "application/json"},
        )
        assert r.status_code == 422, raw
    # Invalid / inverted / empty window.
    assert _vote(client, ["S1", "S2", "S3"], window_start_ns=100,
                 window_end_ns=100).status_code == 422
    assert _vote(client, ["S1", "S2", "S3"], window_start_ns=90,
                 window_end_ns=10).status_code == 422
    assert _vote(client, ["S1", "S2", "S3"], window_start_ns=-1,
                 window_end_ns=10).status_code == 422
    # Empty / missing fields and extra fields.
    assert client.post(
        "/api/v1/batches/B1/redundancy-vote", json={"window_start_ns": 0}
    ).status_code == 422
    assert _vote(client, ["S1", "S2", "S3"], extra=1).status_code == 422
    # Empty expected unit.
    assert _vote(client, ["S1", "S2", "S3"], expected_unit="").status_code == 422


def test_quorum_query_is_read_only(client):
    _post(client, _event("r1", "ROOT", 0, 100, value=1.0, sensor="S1"))
    _post(client, _event("o", "ghost", 20, 80, value=99.0, sensor="S1"))
    _post(client, _event("r2", "ROOT", 0, 100, value=1.1, sensor="S2"))
    before = client.get("/api/v1/batches/B1/result").json()
    _vote(client, ["S1", "S2"])
    after = client.get("/api/v1/batches/B1/result").json()
    assert before == after


def test_quorum_deterministic_across_arrival_order(client):
    events = [
        _event("r1", "ROOT", 0, 100, value=10.0, sensor="S1"),
        _event("o", "ghost", 40, 60, value=50.0, sensor="S1"),
        _event("r2", "ROOT", 0, 100, value=10.2, sensor="S2"),
        _event("r3", "ROOT", 0, 100, value=10.4, sensor="S3"),
    ]

    def run(order):
        client.post("/admin/reset")
        for event in order:
            assert _post(client, event).status_code == 200
        return _vote(client, ["S1", "S2", "S3"]).json()

    first = run(events)
    assert run(list(reversed(events))) == first
    assert run([events[1], events[2], events[0], events[3]]) == first


def test_quorum_half_open_window_is_clipped_and_partitioned(client):
    _post(client, _event("r1", "ROOT", 0, 100, value=1.0, sensor="S1"))
    _post(client, _event("o", "ghost", 40, 60, value=9.0, sensor="S1"))
    _post(client, _event("r2", "ROOT", 0, 100, value=9.0, sensor="S2"))
    body = _vote(
        client, ["S1", "S2"], window_start_ns=10, window_end_ns=90
    ).json()
    spans = body["intervals"]
    assert spans[0]["start_ns"] == 10 and spans[-1]["end_ns"] == 90
    for a, b in zip(spans, spans[1:]):
        assert a["end_ns"] == b["start_ns"]
    assert sum(i["duration_ns"] for i in spans) == 80
    assert [(i["start_ns"], i["end_ns"]) for i in spans] == [
        (10, 40), (40, 60), (60, 90)
    ]
    # A blocked revision abutting the window end does not leak outside it.
    assert spans[-1]["status"] == "rejected"


def test_quorum_adjacent_identical_evidence_merges(client):
    # S1 changes the winning revision (but not the effective vote) at t=50.
    _post(client, _event("root", "ROOT", 0, 100, value=1.0, sensor="S1"))
    _post(client, _event("child", "root", 50, 100, value=1.0, sensor="S1"))
    _post(client, _event("r2", "ROOT", 0, 100, value=1.05, sensor="S2"))
    body = _vote(client, ["S1", "S2"], max_value_diff=0.1).json()
    assert [(i["start_ns"], i["end_ns"], i["status"]) for i in body["intervals"]] == [
        (0, 100, "qualified")
    ]


def test_quorum_openapi_lists_endpoint(client):
    spec = client.get("/openapi.json").json()
    path = spec["paths"]["/api/v1/batches/{batch_id}/redundancy-vote"]["post"]
    ref = path["responses"]["200"]["content"]["application/json"]["schema"]["$ref"]
    assert ref.endswith("/RedundancyVoteResponse")
    props = spec["components"]["schemas"]["RedundancyVoteRequest"]["properties"]
    assert {
        "window_start_ns", "window_end_ns", "sensor_ids",
        "expected_unit", "quorum", "max_value_diff",
    } <= set(props)
