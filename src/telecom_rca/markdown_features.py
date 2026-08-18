"""Structured evidence for the unseen nine-cause Markdown telecom format."""

from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd

from .parser import ParsedQuestion


MARKDOWN_CAUSES = {
    "overlap": "RF or power parameters cause severe overlap coverage",
    "inter_frequency_threshold": "Inter-frequency handover threshold configuration unreasonable",
    "capacity": "Network capacity insufficient or load imbalance between cells",
    "transport": "Test server or transport anomaly causes insufficient upstream traffic",
    "missing_neighbor": "Missing neighbor cell configuration",
    "weak_coverage": "RF, power parameters or site construction cause weak coverage",
    "intra_frequency_high": "Intra-frequency handover threshold too high",
    "intra_frequency_low": "Intra-frequency handover threshold too low",
    "pdcch": "PDCCH resource management parameters unreasonable",
}


def markdown_option_map(question: str) -> dict[str, str]:
    """Map nine stable semantic causes to their shuffled displayed labels."""
    from .options import parse_options

    by_description = {option.description.lower(): option.label for option in parse_options(question)}
    mapping = {
        semantic: by_description[description.lower()]
        for semantic, description in MARKDOWN_CAUSES.items()
        if description.lower() in by_description
    }
    if set(mapping) != set(MARKDOWN_CAUSES):
        missing = sorted(set(MARKDOWN_CAUSES) - set(mapping))
        raise ValueError(f"Markdown option taxonomy is incomplete: {missing}")
    return mapping


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column].replace({"-": np.nan, "": np.nan}), errors="coerce")


def _mean(series: pd.Series) -> float:
    return float(series.mean()) if series.notna().any() else math.nan


def _maximum(series: pd.Series) -> float:
    return float(series.max()) if series.notna().any() else math.nan


def _minimum(series: pd.Series) -> float:
    return float(series.min()) if series.notna().any() else math.nan


def _fraction(mask: pd.Series, valid: pd.Series) -> float:
    valid = valid.fillna(False)
    return float(mask[valid].mean()) if valid.any() else math.nan


def _configured_neighbors(configuration: pd.DataFrame) -> dict[int, set[int]]:
    result: dict[int, set[int]] = {}
    pci = _numeric(configuration, "PCI")
    column = "Neighbor(gNodeB_Freq_PCI)"
    if column not in configuration:
        return result
    for index in configuration.index:
        if pd.isna(pci[index]):
            continue
        entries = re.findall(r"_(\d+)\s*(?:,|\])", str(configuration.at[index, column]))
        result[int(pci[index])] = {int(value) for value in entries}
    return result


def _event_counts(signaling: pd.DataFrame) -> dict[str, int]:
    if "Event Name" not in signaling:
        return {}
    return signaling["Event Name"].astype(str).value_counts().to_dict()


def extract_markdown_evidence(parsed: ParsedQuestion) -> dict[str, float]:
    """Extract radio, configuration, topology, and signaling evidence."""
    required = {"drive_test", "engineering", "configuration", "signaling"}
    if not required.issubset(parsed.tables):
        raise ValueError(f"Markdown evidence requires tables: {sorted(required)}")
    drive = parsed.tables["drive_test"].frame
    config = parsed.tables["configuration"].frame
    signaling = parsed.tables["signaling"].frame

    throughput = _numeric(drive, "Throughput(Mbps)")
    rsrp = _numeric(drive, "Serving RSRP(dBm)")
    sinr = _numeric(drive, "Serving SINR(dB)")
    serving_pci = _numeric(drive, "Serving PCI")
    serving_arfcn = _numeric(drive, "Serving ARFCN")
    rb = _numeric(drive, "RB/slot")
    cce = _numeric(drive, "CCE Fail Rate")
    grant = _numeric(drive, "Grant")
    mcs = _numeric(drive, "Avg MCS")
    initial_bler = _numeric(drive, "Initial BLER(%)")
    residual_bler = _numeric(drive, "Residual BLER(%)")
    low = throughput.lt(100) & throughput.notna()
    if not low.any() and throughput.notna().any():
        low = throughput.eq(throughput.min())

    neighbor_pcis = [_numeric(drive, f"Neighbor {rank} PCI") for rank in range(1, 4)]
    neighbor_rsrps = [_numeric(drive, f"Neighbor {rank} RSRP(dBm)") for rank in range(1, 4)]
    best_neighbor = pd.concat(neighbor_rsrps, axis=1).max(axis=1)
    margin = best_neighbor - rsrp
    signal_valid = best_neighbor.notna() & rsrp.notna()

    configured = _configured_neighbors(config)
    observed_count = pd.Series(0.0, index=drive.index)
    missing_count = pd.Series(0.0, index=drive.index)
    for index in drive.index:
        if pd.isna(serving_pci[index]):
            continue
        allowed = configured.get(int(serving_pci[index]), set())
        for series in neighbor_pcis:
            if pd.notna(series[index]):
                observed_count[index] += 1
                if int(series[index]) not in allowed:
                    missing_count[index] += 1
    neighbor_valid = observed_count.gt(0)

    a2 = _numeric(config, "CovInterFreqA2RsrpThld(dBm)")
    a5_1 = _numeric(config, "CovInterFreqA5RsrpThld1(dBm)")
    a5_2 = _numeric(config, "CovInterFreqA5RsrpThld2(dBm)")
    a3_raw = _numeric(config, "IntraFreqHoA3Offset(0.5dB)")
    pdcch_text = config.get("PdcchOccupiedSymbolNum", pd.Series(index=config.index, dtype=str))
    pdcch_symbols = pd.to_numeric(
        pdcch_text.astype(str).str.extract(r"(\d+)", expand=False), errors="coerce"
    )
    events = _event_counts(signaling)

    return {
        "drive_rows": float(len(drive)),
        "low_row_count": float(low.sum()),
        "low_fraction": float(low.mean()),
        "throughput_min": _minimum(throughput),
        "low_throughput_mean": _mean(throughput[low]),
        "low_rsrp_mean": _mean(rsrp[low]),
        "low_rsrp_min": _minimum(rsrp[low]),
        "low_sinr_mean": _mean(sinr[low]),
        "low_sinr_min": _minimum(sinr[low]),
        "low_weak_rsrp_fraction": _fraction(rsrp.lt(-105) & low, rsrp.notna() & low),
        "low_negative_sinr_fraction": _fraction(sinr.lt(0) & low, sinr.notna() & low),
        "low_overlap_3db_fraction": _fraction(margin.ge(-3) & low, signal_valid & low),
        "low_neighbor_stronger_fraction": _fraction(margin.gt(0) & low, signal_valid & low),
        "low_neighbor_margin_max_db": _maximum(margin[low]),
        "low_rb_mean": _mean(rb[low]),
        "low_rb_below_100_fraction": _fraction(rb.lt(100) & low, rb.notna() & low),
        "low_cce_fail_mean": _mean(cce[low]),
        "low_cce_fail_max": _maximum(cce[low]),
        "low_grant_mean": _mean(grant[low]),
        "low_grant_min": _minimum(grant[low]),
        "low_mcs_mean": _mean(mcs[low]),
        "low_initial_bler_mean": _mean(initial_bler[low]),
        "low_residual_bler_mean": _mean(residual_bler[low]),
        "serving_pci_count": float(serving_pci.nunique()),
        "serving_arfcn_count": float(serving_arfcn.nunique()),
        "missing_neighbor_fraction": float(
            (missing_count[neighbor_valid] / observed_count[neighbor_valid]).mean()
        ) if neighbor_valid.any() else math.nan,
        "rows_with_missing_neighbor_fraction": float(
            missing_count[neighbor_valid].gt(0).mean()
        ) if neighbor_valid.any() else math.nan,
        "a2_threshold_max_dbm": _maximum(a2),
        "a5_threshold1_max_dbm": _maximum(a5_1),
        "a5_threshold2_max_dbm": _maximum(a5_2),
        "a3_offset_min_db": _minimum(a3_raw) * 0.5,
        "a3_offset_max_db": _maximum(a3_raw) * 0.5,
        "a3_offset_mean_db": _mean(a3_raw) * 0.5,
        "pdcch_symbols_mean": _mean(pdcch_symbols),
        "event_a2_count": float(events.get("NREventA2", 0)),
        "event_a3_count": float(events.get("NREventA3", 0)),
        "event_a5_count": float(events.get("NREventA5", 0)),
        "handover_attempt_count": float(events.get("NRHandoverAttempt", 0)),
        "rrc_reestablish_count": float(events.get("NRRRCReestablishAttempt", 0)),
        "random_access_attempt_count": float(events.get("NRRandomAccessAttempt", 0)),
    }


def evidence_flags(features: dict[str, float]) -> dict[str, bool]:
    """Domain-readable anomaly flags; these are evidence, not predictions."""
    return {
        "strong_overlap": features["low_overlap_3db_fraction"] >= 0.5,
        "weak_coverage": (
            features["low_weak_rsrp_fraction"] > 0
            or features["low_negative_sinr_fraction"] > 0
        ),
        "inter_frequency_threshold_suspicious": features["a2_threshold_max_dbm"] > -100,
        "intra_frequency_threshold_high": features["a3_offset_max_db"] >= 5,
        "intra_frequency_threshold_low": features["a3_offset_min_db"] <= 1,
        "missing_neighbor": features["rows_with_missing_neighbor_fraction"] > 0,
        "pdcch_stress": features["low_cce_fail_max"] >= 0.4,
        "low_scheduling": features["low_rb_below_100_fraction"] >= 0.5,
        "rrc_instability": features["rrc_reestablish_count"] > 0,
    }


def compact_evidence_summary(features: dict[str, float]) -> str:
    """Render bounded evidence for a later Qwen prompt without raw tables."""
    flags = evidence_flags(features)
    active = ", ".join(name for name, value in flags.items() if value) or "none"
    return (
        f"Low-throughput rows={features['low_row_count']:.0f}; "
        f"RSRP mean/min={features['low_rsrp_mean']:.1f}/{features['low_rsrp_min']:.1f} dBm; "
        f"SINR mean/min={features['low_sinr_mean']:.1f}/{features['low_sinr_min']:.1f} dB; "
        f"neighbor-within-3dB fraction={features['low_overlap_3db_fraction']:.2f}; "
        f"RB mean={features['low_rb_mean']:.1f}; CCE fail max={features['low_cce_fail_max']:.2f}; "
        f"missing-neighbor rows={features['rows_with_missing_neighbor_fraction']:.2f}; "
        f"A2 max={features['a2_threshold_max_dbm']:.0f} dBm; "
        f"A3 range={features['a3_offset_min_db']:.1f}-{features['a3_offset_max_db']:.1f} dB; "
        f"A2/A3/A5 events={features['event_a2_count']:.0f}/"
        f"{features['event_a3_count']:.0f}/{features['event_a5_count']:.0f}; "
        f"handover attempts={features['handover_attempt_count']:.0f}; "
        f"RRC reestablish={features['rrc_reestablish_count']:.0f}; flags={active}."
    )


def evidence_scores(features: dict[str, float]) -> dict[str, float]:
    """Return conservative evidence strengths, not calibrated probabilities."""
    good_rf = features["low_rsrp_mean"] > -95 and features["low_sinr_mean"] > 8
    scores = {
        "overlap": max(0.0, features["low_overlap_3db_fraction"]) * 2.0,
        "inter_frequency_threshold": 3.0 if features["a2_threshold_max_dbm"] > -100 else 0.0,
        "capacity": 2.0 if features["low_rb_mean"] > 230 else 0.0,
        "transport": 2.5 if good_rf and features["low_rb_below_100_fraction"] >= 0.5 else 0.0,
        "missing_neighbor": max(0.0, features["rows_with_missing_neighbor_fraction"]) * 3.0,
        "weak_coverage": (
            max(0.0, features["low_weak_rsrp_fraction"])
            + max(0.0, features["low_negative_sinr_fraction"])
        ) * 2.0,
        "intra_frequency_high": 3.0 if features["a3_offset_max_db"] >= 5 else 0.0,
        "intra_frequency_low": 3.0 if features["a3_offset_min_db"] <= 1 else 0.0,
        "pdcch": max(0.0, features["low_cce_fail_max"]) * 5.0,
    }
    return scores


def evidence_hypothesis(features: dict[str, float]) -> tuple[str, float, float]:
    """Return top cause, top score, and margin over second place."""
    ordered = sorted(evidence_scores(features).items(), key=lambda item: (-item[1], item[0]))
    top_cause, top_score = ordered[0]
    second_score = ordered[1][1]
    return top_cause, top_score, top_score - second_score


def evidence_guided_messages(
    question: str,
    features: dict[str, float],
) -> list[dict[str, str]]:
    """Create a compact nine-choice prompt from independently extracted evidence."""
    from .options import parse_options

    options = parse_options(question)
    if len(options) != 9:
        raise ValueError(f"Expected nine Markdown options, found {len(options)}")
    option_text = "\n".join(f"{option.label}: {option.description}" for option in options)
    example = options[0].label
    return [
        {
            "role": "system",
            "content": (
                "You are a 5G radio network troubleshooting expert. Select the single root cause "
                "most directly supported by the structured evidence. Return exactly one boxed "
                "choice and no explanation."
            ),
        },
        {
            "role": "user",
            "content": (
                "Structured diagnostic evidence:\n"
                + compact_evidence_summary(features)
                + "\n\nCandidate root causes:\n"
                + option_text
                + f"\n\nYour entire response must be exactly one boxed choice, for example: \\boxed{{{example}}}"
            ),
        },
    ]
