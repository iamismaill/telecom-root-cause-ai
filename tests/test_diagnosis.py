import pytest

from telecom_rca.diagnosis import diagnose_standard


BASE = {
    "low_speed_max": 30.0,
    "low_distance_max_km": 0.2,
    "low_rbs_mean": 200.0,
    "ping_pong_count": 0.0,
    "handover_count": 1.0,
    "low_mod30_conflict_fraction": 0.0,
    "low_close_noncolocated_fraction": 0.0,
    "low_tilt_excess_max_deg": 0.0,
}


@pytest.mark.parametrize(
    ("updates", "label"),
    [
        ({"low_speed_max": 41}, "C7"),
        ({"low_distance_max_km": 1.01}, "C2"),
        ({"low_rbs_mean": 159}, "C8"),
        ({"ping_pong_count": 1}, "C5"),
        ({"low_mod30_conflict_fraction": 1}, "C6"),
        ({"low_close_noncolocated_fraction": 1}, "C4"),
        ({"low_tilt_excess_max_deg": 5.1}, "C1"),
        ({}, "C3"),
    ],
)
def test_each_rule(updates: dict[str, float], label: str) -> None:
    assert diagnose_standard({**BASE, **updates}).label == label

