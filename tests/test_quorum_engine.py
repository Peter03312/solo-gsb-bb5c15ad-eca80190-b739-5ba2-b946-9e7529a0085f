"""Engine unit tests: redundant-sensor trusted-majority voting sweep."""

from app.engine import evaluate_redundancy_vote
from app.storage import BatchStore, StoredEvent


def _ev(rev, sensor, parent, start=0, end=10, value=1.0, unit="u",
        withdrawn=False, batch="B", key=None):
    return StoredEvent(
        batch_id=batch,
        sensor_id=sensor,
        revision_id=rev,
        parent_id=parent,
        idempotency_key=key or f"k-{sensor}-{rev}",
        start_ns=start,
        end_ns=end,
        value=None if withdrawn else float(value),
        unit=None if withdrawn else unit,
        withdrawn=withdrawn,
    )


def _store(events):
    store = BatchStore()
    for e in events:
        store.events[e.revision_id] = e
        store.idempotency[e.idempotency_key] = "x"
    return store


def _vote(store, sensors, quorum=2, diff=0.5, start=0, end=100, unit="u"):
    return evaluate_redundancy_vote(
        "B", start, end, list(sensors), unit, quorum, diff, store
    )


def _spans(doc):
    return [(i["start_ns"], i["end_ns"], i["status"]) for i in doc["intervals"]]


def test_quorum_full_agreement_qualifies_with_all_members():
    store = _store([
        _ev("r1", "S1", "ROOT", 0, 100, value=10.0),
        _ev("r2", "S2", "ROOT", 0, 100, value=10.3),
        _ev("r3", "S3", "ROOT", 0, 100, value=10.5),
    ])
    doc = _vote(store, ["S1", "S2", "S3"], quorum=2, diff=0.5)
    assert doc["status"] == "qualified"
    (span,) = doc["intervals"]
    assert (span["start_ns"], span["end_ns"], span["duration_ns"]) == (0, 100, 100)
    assert span["status"] == "qualified"
    assert [m["sensor_id"] for m in span["members"]] == ["S1", "S2", "S3"]
    assert span["non_votes"] == []
    assert span["missing_ancestors"] == []
    assert span["largest_compatible_size"] == 3 and span["blocked_count"] == 0
    assert len(doc["input_hash"]) == 64


def test_quorum_spread_boundary_is_inclusive():
    store = _store([
        _ev("r1", "S1", "ROOT", 0, 100, value=10.0),
        _ev("r2", "S2", "ROOT", 0, 100, value=10.5),
    ])
    assert _vote(store, ["S1", "S2"], quorum=2, diff=0.5)["status"] == "qualified"
    doc = _vote(store, ["S1", "S2"], quorum=2, diff=0.499)
    assert doc["status"] == "rejected"
    assert doc["intervals"][0]["largest_compatible_size"] == 1


def test_quorum_chooses_largest_compatible_set_then_lexicographic():
    # Two disjoint value clusters of size two; q=2 makes either cluster a
    # valid largest set, and the lexicographically smallest sorted id tuple
    # breaks the tie.
    store = _store([
        _ev("r1", "S1", "ROOT", 0, 50, value=1.0),
        _ev("r2", "S2", "ROOT", 0, 50, value=1.1),
        _ev("r3", "S3", "ROOT", 0, 50, value=5.0),
        _ev("r4", "S4", "ROOT", 0, 50, value=5.1),
    ])
    doc = _vote(store, ["S1", "S2", "S3", "S4"], quorum=2, diff=0.2, end=50)
    # All values are constant on the span; the lexicographically smallest
    # largest set wins.
    assert _spans(doc) == [(0, 50, "qualified")]
    span = doc["intervals"][0]
    assert span["status"] == "qualified"
    assert [m["sensor_id"] for m in span["members"]] == ["S1", "S2"]
    assert span["largest_compatible_size"] == 2


def test_quorum_lexicographic_tie_among_three_sensor_sets():
    store = _store([
        _ev("a", "S1", "ROOT", 0, 50, value=1.0),
        _ev("b", "S2", "ROOT", 0, 50, value=1.0),
        _ev("c", "S3", "ROOT", 0, 50, value=1.0),
        _ev("d", "S4", "ROOT", 0, 50, value=9.0),
        _ev("e", "S5", "ROOT", 0, 50, value=9.0),
    ])
    doc = _vote(store, ["S1", "S2", "S3", "S4", "S5"], quorum=3, diff=0.0)
    assert [m["sensor_id"] for m in doc["intervals"][0]["members"]] == [
        "S1", "S2", "S3"
    ]


def test_quorum_blocked_masks_grounded_value_and_is_pending():
    # S1 has a grounded 10.0 but a blocked revision covers [40,60); S2=10.1
    # agrees elsewhere. With q=2 the blocked span is pending backfill.
    store = _store([
        _ev("root", "S1", "ROOT", 0, 100, value=10.0),
        _ev("orphan", "S1", "ghost", 40, 60, value=99.0),
        _ev("r2", "S2", "ROOT", 0, 100, value=10.1),
    ])
    doc = _vote(store, ["S1", "S2"], quorum=2, diff=0.5)
    assert doc["status"] == "qualified"
    assert _spans(doc) == [
        (0, 40, "qualified"),
        (40, 60, "pending_backfill"),
        (60, 100, "qualified"),
    ]
    pending = doc["intervals"][1]
    (nv,) = pending["non_votes"]
    assert nv["sensor_id"] == "S1" and nv["reason"] == "blocked"
    assert nv["missing_ancestors"] == ["ghost"]
    assert pending["missing_ancestors"] == ["ghost"]
    assert pending["largest_compatible_size"] == 1
    assert pending["blocked_count"] == 1
    assert pending["members"] == []


def test_quorum_blocked_can_be_rejected_when_even_backfill_cannot_reach_q():
    # q=3 of 3; only one vote (S1), S2 blocked, S3 uncovered: 1+1 < 3.
    store = _store([
        _ev("r1", "S1", "ROOT", 0, 100, value=10.0),
        _ev("o2", "S2", "ghost", 0, 100, value=5.0),
        # S3 exists as a sensor but has no grounded coverage of the window.
        _ev("r3", "S3", "ROOT", 200, 300, value=1.0),
    ])
    doc = _vote(store, ["S1", "S2", "S3"], quorum=3, diff=0.5)
    (span,) = doc["intervals"]
    assert span["status"] == "rejected"
    assert span["missing_ancestors"] == []
    reasons = {n["sensor_id"]: n for n in span["non_votes"]}
    assert set(reasons) == {"S2", "S3"}
    assert reasons["S2"]["reason"] == "blocked"
    assert reasons["S2"]["missing_ancestors"] == ["ghost"]
    assert reasons["S3"]["reason"] == "uncovered"
    assert reasons["S3"]["value"] is None and reasons["S3"]["unit"] is None
    assert span["blocked_count"] == 1


def test_quorum_blocked_does_not_mask_when_outside_window():
    store = _store([
        _ev("root", "S1", "ROOT", 0, 100, value=10.0),
        _ev("orphan", "S1", "ghost", 80, 120, value=99.0),
        _ev("r2", "S2", "ROOT", 0, 100, value=10.1),
    ])
    doc = _vote(store, ["S1", "S2"], quorum=2, diff=0.5, start=0, end=80)
    assert _spans(doc) == [(0, 80, "qualified")]


def test_quorum_non_vote_reasons_withdrawn_uncovered_unit_mismatch():
    store = _store([
        _ev("root", "S1", "ROOT", 0, 80, value=1.0),
        _ev("w", "S1", "root", 20, 40, withdrawn=True),
        _ev("f", "S1", "w", 40, 60, value=1.0, unit="F"),
        _ev("r2", "S2", "ROOT", 0, 100, value=1.0),
        _ev("r3", "S3", "ROOT", 0, 100, value=1.0),
    ])
    doc = _vote(store, ["S1", "S2", "S3"], quorum=3, diff=0.5)
    reasons = {
        (lo, hi): {n["sensor_id"]: n for n in span["non_votes"]}
        for lo, hi, span in [
            (i["start_ns"], i["end_ns"], i) for i in doc["intervals"]
        ]
    }
    assert reasons[(0, 20)] == {}
    assert reasons[(20, 40)]["S1"]["reason"] == "withdrawn"
    unit_nv = reasons[(40, 60)]["S1"]
    assert unit_nv["reason"] == "unit_mismatch"
    assert unit_nv["unit"] == "F" and unit_nv["value"] == 1.0
    assert reasons[(60, 80)] == {}
    assert reasons[(80, 100)]["S1"]["reason"] == "uncovered"


def test_quorum_invalid_cycle_revision_does_not_participate():
    store = _store([
        _ev("c1", "S1", "c2", 0, 100, value=99.0),
        _ev("c2", "S1", "c1", 0, 100, value=99.0),
        _ev("r2", "S2", "ROOT", 0, 100, value=1.0),
        _ev("r3", "S3", "ROOT", 0, 100, value=1.0),
    ])
    doc = _vote(store, ["S1", "S2", "S3"], quorum=2, diff=0.5)
    (span,) = doc["intervals"]
    # S1 is uncovered (cycles excluded), S2/S3 agree -> qualified.
    assert span["status"] == "qualified"
    assert [m["sensor_id"] for m in span["members"]] == ["S2", "S3"]
    assert span["non_votes"][0] == {
        "sensor_id": "S1",
        "reason": "uncovered",
        "value": None,
        "unit": None,
        "missing_ancestors": [],
    }


def test_quorum_window_clipping_and_half_open_boundaries():
    store = _store([
        _ev("r1", "S1", "ROOT", 0, 100, value=1.0),
        _ev("r2", "S2", "ROOT", 0, 100, value=1.1),
    ])
    doc = _vote(store, ["S1", "S2"], quorum=2, diff=0.5, start=10, end=90)
    assert _spans(doc) == [(10, 90, "qualified")]
    # Window fully outside coverage: both uncovered -> rejected.
    outside = _vote(store, ["S1", "S2"], quorum=2, diff=0.5, start=200, end=300)
    assert _spans(outside) == [(200, 300, "rejected")]
    assert len(outside["intervals"][0]["non_votes"]) == 2


def test_quorum_intervals_partition_the_whole_window():
    store = _store([
        _ev("root", "S1", "ROOT", 0, 100, value=1.0),
        _ev("o1", "S1", "g1", 20, 40, value=9.0),
        _ev("r3", "S3", "ROOT", 0, 40, value=1.1),
        _ev("r2", "S2", "ROOT", 0, 100, value=9.0),
    ])
    doc = _vote(store, ["S1", "S2", "S3"], quorum=2, diff=0.5, start=5, end=95)
    spans = doc["intervals"]
    assert spans[0]["start_ns"] == 5 and spans[-1]["end_ns"] == 95
    for a, b in zip(spans, spans[1:]):
        assert a["end_ns"] == b["start_ns"]
    assert sum(i["end_ns"] - i["start_ns"] for i in spans) == 90
    assert [(i["start_ns"], i["end_ns"]) for i in spans] == [
        (5, 20), (20, 40), (40, 95)
    ]
    assert [i["status"] for i in spans] == [
        "qualified", "pending_backfill", "rejected"
    ]


def test_quorum_adjacent_intervals_merge_only_on_identical_evidence():
    # q=3 of 3: on the blocked spans only S2 votes (S1 masked, S3 has no
    # coverage there): 1 compatible vote + 1 blocked sensor < 3 but could
    # reach it only after also covering S3, so the spans are rejected.
    # The two blocked spans carry different missing ancestors, so even
    # identical verdicts stay split; [60,100) with S1+S2+S3 all at 9.0 is
    # qualified and merges across internal fold boundaries.
    store = _store([
        _ev("root", "S1", "ROOT", 0, 100, value=9.0),
        _ev("b1", "S1", "g1", 0, 30, value=9.0),
        _ev("b2", "S1", "g2", 30, 60, value=9.0),
        _ev("r2", "S2", "ROOT", 0, 100, value=9.0),
        _ev("r3", "S3", "ROOT", 60, 100, value=9.0),
    ])
    doc = _vote(store, ["S1", "S2", "S3"], quorum=3, diff=0.0)
    rejected = doc["intervals"][:2]
    assert [(i["start_ns"], i["end_ns"], i["status"]) for i in rejected] == [
        (0, 30, "rejected"),
        (30, 60, "rejected"),
    ]
    nv0 = {n["sensor_id"]: n for n in rejected[0]["non_votes"]}
    nv1 = {n["sensor_id"]: n for n in rejected[1]["non_votes"]}
    assert nv0["S1"]["missing_ancestors"] == ["g1"]
    assert nv1["S1"]["missing_ancestors"] == ["g2"]
    # Both rejected spans list S3 uncovered; the differing ancestor still
    # forbids merging.
    assert nv0["S3"]["reason"] == "uncovered" and nv1["S3"]["reason"] == "uncovered"
    assert [(i["start_ns"], i["end_ns"]) for i in doc["intervals"]
            if i["status"] == "qualified"] == [(60, 100)]


def test_quorum_pending_spans_with_different_ancestors_do_not_merge():
    # q=2 of 2: S2 votes at 9.0 throughout; S1 is blocked on [0,60) by g1
    # then g2, so each blocked span is pending (1 vote + 1 blocked >= 2).
    # The missing-ancestor evidence changes at t=30, so the two pending
    # spans must not merge.
    store = _store([
        _ev("root", "S1", "ROOT", 0, 100, value=9.0),
        _ev("b1", "S1", "g1", 0, 30, value=9.0),
        _ev("b2", "S1", "g2", 30, 60, value=9.0),
        _ev("r2", "S2", "ROOT", 0, 100, value=9.0),
    ])
    doc = _vote(store, ["S1", "S2"], quorum=2, diff=0.0)
    assert _spans(doc) == [
        (0, 30, "pending_backfill"),
        (30, 60, "pending_backfill"),
        (60, 100, "qualified"),
    ]
    assert doc["intervals"][0]["missing_ancestors"] == ["g1"]
    assert doc["intervals"][1]["missing_ancestors"] == ["g2"]


def test_quorum_provenance_only_change_merges_adjacent_intervals():
    # S1's value changes winning revision at 50 but the effective vote is
    # identical (1.0/u) and S2 keeps agreeing: one merged qualified span.
    store = _store([
        _ev("root", "S1", "ROOT", 0, 100, value=1.0),
        _ev("child", "S1", "root", 50, 100, value=1.0),
        _ev("r2", "S2", "ROOT", 0, 100, value=1.0),
    ])
    doc = _vote(store, ["S1", "S2"], quorum=2, diff=0.0)
    assert _spans(doc) == [(0, 100, "qualified")]


def test_quorum_different_member_values_split_adjacent_intervals():
    # S2 changes value at 50: chosen members stay {S1,S2} but the member
    # evidence (value) differs, so qualified spans must not merge.
    store = _store([
        _ev("r1", "S1", "ROOT", 0, 100, value=1.0),
        _ev("b", "S2", "ROOT", 0, 50, value=1.05),
        _ev("c", "S2", "b", 50, 100, value=1.15),
    ])
    doc = _vote(store, ["S1", "S2"], quorum=2, diff=0.2)
    assert _spans(doc) == [(0, 50, "qualified"), (50, 100, "qualified")]
    assert doc["intervals"][1]["members"][1]["value"] == 1.15


def test_quorum_overall_status_prefers_qualified_then_pending():
    store = _store([
        _ev("root", "S1", "ROOT", 0, 100, value=1.0),
        _ev("b", "S1", "g", 50, 100, value=9.0),
        _ev("r2", "S2", "ROOT", 0, 100, value=1.1),
    ])
    qualified_and_pending = _vote(store, ["S1", "S2"], quorum=2, diff=0.5)
    assert qualified_and_pending["status"] == "qualified"
    assert _spans(qualified_and_pending) == [
        (0, 50, "qualified"), (50, 100, "pending_backfill")
    ]
    # Shift the blocked span over the whole window -> only pending remains.
    store.events["b"] = _ev("b", "S1", "g", 0, 100, value=9.0)
    only_pending = _vote(store, ["S1", "S2"], quorum=2, diff=0.5)
    assert only_pending["status"] == "pending_backfill"
    assert _spans(only_pending) == [(0, 100, "pending_backfill")]


def test_quorum_hash_stable_and_independent_of_query_params():
    store = _store([
        _ev("r1", "S1", "ROOT", 0, 100, value=1.0),
        _ev("r2", "S2", "ROOT", 0, 100, value=1.1),
        _ev("r3", "S3", "ROOT", 0, 100, value=1.2),
    ])
    h1 = _vote(store, ["S1", "S2", "S3"], quorum=2, diff=0.5)["input_hash"]
    h2 = _vote(store, ["S1", "S2", "S3"], quorum=3, diff=9.0,
               start=10, end=90, unit="other")["input_hash"]
    assert h1 == h2
    store.events["o"] = _ev("o", "S1", "g", 0, 5, value=9.0)
    h3 = _vote(store, ["S1", "S2", "S3"], quorum=2, diff=0.5)["input_hash"]
    assert h3 != h1


def test_quorum_unknown_sensor_returns_none():
    store = _store([_ev("r1", "S1", "ROOT", 0, 10, value=1.0)])
    assert _vote(store, ["S1", "MISSING"], quorum=2) is None
    assert _vote(store, ["S1", "S2", "MISSING"], quorum=2) is None


def test_quorum_deterministic_across_insertion_and_request_order():
    events = [
        _ev("r1", "S1", "ROOT", 0, 100, value=1.0),
        _ev("o", "S1", "g", 40, 60, value=9.0),
        _ev("r2", "S2", "ROOT", 0, 100, value=9.0),
        _ev("r3", "S3", "ROOT", 0, 100, value=1.1),
    ]

    def build(order, sensor_order):
        store = BatchStore()
        for e in order:
            store.events[e.revision_id] = e
        doc = evaluate_redundancy_vote(
            "B", 0, 100, sensor_order, "u", 2, 0.5, store
        )
        doc.pop("input_hash")
        # The request echoes the asked sensor order; normalize it so the
        # comparison isolates order-independence of the verdict itself.
        doc["sensor_ids"] = ["S1", "S2", "S3"]
        return doc

    base = build(events, ["S1", "S2", "S3"])
    assert build(list(reversed(events)), ["S3", "S2", "S1"]) == base
