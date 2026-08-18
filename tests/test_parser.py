import pandas as pd

from telecom_rca.data import load_current_csv
from telecom_rca.parser import parse_question


STANDARD = """Question text
User plane drive test data as follows:
Timestamp|Longitude|Latitude|5G KPI PCell Layer2 MAC DL Throughput [Mbps]
2026-01-01|1.0|2.0|500

Engeneering parameters data as follows:
gNodeB ID|Cell ID|Longitude|Latitude|Mechanical Downtilt|PCI
001|1|1.0|2.0|6|42
"""

MARKDOWN = """**Drive Test Data**
| Time | UE | Throughput(Mbps) | Serving PCI |
|---|---|---:|---:|
| 10:00 | UE1 | 90 | 10 |

**Parameter Data**
| gNodeB ID | Cell ID | Mech Tilt(deg) | PCI |
|---|---:|---:|---:|
| 001 | 1 | 6 | 10 |
"""


def test_standard_tables_are_classified() -> None:
    result = parse_question(STANDARD)
    assert result.format_name == "standard"
    assert result.parsed_cleanly
    assert set(result.tables) == {"drive_test", "engineering"}


def test_markdown_tables_are_classified() -> None:
    result = parse_question(MARKDOWN)
    assert result.format_name == "markdown"
    assert result.parsed_cleanly
    assert len(result.tables["drive_test"].frame) == 1


def test_malformed_rows_are_reported() -> None:
    malformed = STANDARD.replace("2026-01-01|1.0|2.0|500", "2026-01-01|1.0|500")
    result = parse_question(malformed)
    assert result.errors
    assert "expected 4 cells, found 3" in result.errors[0]


def test_all_official_telecom_questions_have_core_tables() -> None:
    expected_telecom = {"train.csv": 2400, "validation_questions.csv": 864, "test.csv": 781}
    for filename, expected in expected_telecom.items():
        frame = load_current_csv(filename)
        results = [parse_question(q) for q in frame["question"].astype(str)]
        telecom = [result for result in results if result.is_telecom]
        assert len(telecom) == expected
        assert all("engineering" in result.tables for result in telecom)
        assert all(not result.errors for result in telecom)

