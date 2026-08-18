import math

import pytest

from telecom_rca.features import extract_diagnostic_features, throughput_threshold
from telecom_rca.data import load_current_csv
from telecom_rca.parser import parse_question


QUESTION = """Analyze throughput dropping below 600Mbps.
Timestamp|Longitude|Latitude|GPS Speed (km/h)|5G KPI PCell RF Serving PCI|5G KPI PCell RF Serving SS-RSRP [dBm]|5G KPI PCell RF Serving SS-SINR [dB]|5G KPI PCell Layer2 MAC DL Throughput [Mbps]|Measurement PCell Neighbor Cell Top Set(Cell Level) Top 1 PCI|Measurement PCell Neighbor Cell Top Set(Cell Level) Top 1 Filtered Tx BRSRP [dBm]|5G KPI PCell Layer1 DL RB Num (Including 0)
t1|0.0000|0.0000|20|31|-90|10|700|61|-95|200
t2|0.0100|0.0000|45|31|-100|2|500|61|-90|100
t3|0.0101|0.0000|50|61|-98|3|400|31|-85|120
t4|0.0102|0.0000|30|31|-95|5|450|61|-88|130

gNodeB ID|Cell ID|Longitude|Latitude|Mechanical Downtilt|Digital Tilt|Height|PCI
A|1|0.0000|0.0000|10|255|30|31
B|1|0.0100|0.0000|5|5|25|61
"""


def test_threshold_is_read_from_question() -> None:
    assert throughput_threshold(QUESTION) == 600
    assert throughput_threshold("Throughput drops below 100 Mbps") == 100


def test_features_focus_on_low_throughput_rows() -> None:
    features = extract_diagnostic_features(QUESTION, parse_question(QUESTION))
    assert features["low_row_count"] == 3
    assert features["speed_max"] == 50
    assert features["low_speed_max"] == 50
    assert features["low_rbs_mean"] == pytest.approx((100 + 120 + 130) / 3)
    assert features["handover_count"] == 2
    assert features["ping_pong_count"] == 1
    assert features["low_mod30_conflict_fraction"] == 1
    assert features["low_neighbor_stronger_fraction"] == 1
    assert features["low_serving_total_tilt_max"] == 16
    assert features["low_distance_max_km"] > 1


def test_non_telecom_feature_extraction_is_rejected() -> None:
    text = "What is 2 + 2? 1: 3 2: 4"
    with pytest.raises(ValueError):
        extract_diagnostic_features(text, parse_question(text))


def test_features_cover_all_official_telecom_formats() -> None:
    for filename in ("train.csv", "validation_questions.csv", "test.csv"):
        frame = load_current_csv(filename)
        parsed_rows = [(question, parse_question(question)) for question in frame["question"].astype(str)]
        telecom_rows = [(question, parsed) for question, parsed in parsed_rows if parsed.is_telecom]
        features = [extract_diagnostic_features(question, parsed) for question, parsed in telecom_rows]
        assert len(features) == len(telecom_rows)
        assert all(feature["row_count"] > 0 for feature in features)
