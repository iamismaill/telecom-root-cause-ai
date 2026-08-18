from telecom_rca.options import map_standard_cause, parse_options


QUESTION = """Choose one:
3: Frequent handovers degrade performance.
1: Neighbor cell and serving cell have the same PCI mod 30, leading to interference.
8: Non-colocated co-frequency neighboring cells cause severe overlapping coverage.
2: Test vehicle speed exceeds 40km/h, impacting user throughput.
4: Average scheduled RBs are below 160, affecting throughput.
5: A neighboring cell provides higher throughput.
7: The serving cell's coverage distance exceeds 1km, resulting in over-shooting.
6: The serving cell's downtilt angle is too large, causing weak coverage at the far end.
"""


def test_options_are_parsed_in_displayed_order() -> None:
    assert [option.label for option in parse_options(QUESTION)] == ["3", "1", "8", "2", "4", "5", "7", "6"]


def test_every_semantic_cause_maps_after_shuffle() -> None:
    expected = {"C1": "6", "C2": "7", "C3": "5", "C4": "8", "C5": "3", "C6": "1", "C7": "2", "C8": "4"}
    assert {label: map_standard_cause(QUESTION, label) for label in expected} == expected

