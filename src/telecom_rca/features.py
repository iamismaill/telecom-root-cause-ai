"""Independent, interval-aware diagnostic feature extraction.

Features are calculated from parsed official challenge records. The degraded
interval is defined by the throughput threshold stated in each question, so
healthy rows do not dominate evidence for the root cause.
"""

from __future__ import annotations

import math
import re
from typing import Iterable

import numpy as np
import pandas as pd

from .parser import ParsedQuestion


def _normal(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def _column(frame: pd.DataFrame, *aliases: str) -> str | None:
    normalized = {_normal(col): col for col in frame.columns}
    for alias in aliases:
        if _normal(alias) in normalized:
            return normalized[_normal(alias)]
    return None


def _numeric(frame: pd.DataFrame, *aliases: str) -> pd.Series:
    name = _column(frame, *aliases)
    if name is None:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[name].replace({"-": np.nan, "": np.nan}), errors="coerce")


def _safe_mean(values: pd.Series) -> float:
    return float(values.mean()) if values.notna().any() else math.nan


def _safe_max(values: pd.Series) -> float:
    return float(values.max()) if values.notna().any() else math.nan


def _safe_min(values: pd.Series) -> float:
    return float(values.min()) if values.notna().any() else math.nan


def _fraction(mask: pd.Series, eligible: pd.Series | None = None) -> float:
    valid = eligible if eligible is not None else mask.notna()
    valid = valid.fillna(False)
    return float(mask[valid].mean()) if valid.any() else math.nan


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlat = p2 - p1
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlon / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(a))


def throughput_threshold(question: str, default: float = 600.0) -> float:
    """Extract the throughput threshold stated in Mbps."""
    patterns = (
        r"throughput\s+(?:dropping|drops?)\s+below\s+([0-9]+(?:\.[0-9]+)?)\s*mbps",
        r"below\s+([0-9]+(?:\.[0-9]+)?)\s*mbps",
    )
    lowered = question.lower()
    for pattern in patterns:
        match = re.search(pattern, lowered)
        if match:
            return float(match.group(1))
    return default


def _engineering_map(frame: pd.DataFrame) -> dict[int, dict[str, object]]:
    pci = _numeric(frame, "PCI")
    lat = _numeric(frame, "Latitude")
    lon = _numeric(frame, "Longitude")
    mech = _numeric(frame, "Mechanical Downtilt", "Mech Tilt(deg)")
    digital = _numeric(frame, "Digital Tilt", "Elec Tilt(deg)")
    height = _numeric(frame, "Height", "Ant Height(m)")
    gnode_col = _column(frame, "gNodeB ID")
    beam_col = _column(frame, "Beam Scenario")
    result: dict[int, dict[str, object]] = {}
    for index in frame.index:
        if pd.isna(pci[index]):
            continue
        digital_value = digital[index]
        if pd.notna(digital_value) and digital_value == 255:
            digital_value = 6.0
        total_tilt = (
            float(mech[index]) + float(digital_value)
            if pd.notna(mech[index]) and pd.notna(digital_value)
            else math.nan
        )
        result[int(pci[index])] = {
            "lat": float(lat[index]) if pd.notna(lat[index]) else math.nan,
            "lon": float(lon[index]) if pd.notna(lon[index]) else math.nan,
            "mechanical_tilt": float(mech[index]) if pd.notna(mech[index]) else math.nan,
            "digital_tilt": float(digital_value) if pd.notna(digital_value) else math.nan,
            "total_tilt": total_tilt,
            "height": float(height[index]) if pd.notna(height[index]) else math.nan,
            "gnodeb": str(frame.at[index, gnode_col]).strip() if gnode_col else None,
            "beam_scenario": str(frame.at[index, beam_col]).strip() if beam_col else None,
        }
    return result


def _vertical_beamwidth(scenario: object) -> float:
    """Return vertical beamwidth from the relationship stated in the prompt."""
    text = str(scenario or "DEFAULT").upper()
    match = re.search(r"SCENARIO[_ ]?(\d+)", text)
    if not match:
        return 6.0
    number = int(match.group(1))
    if number <= 5:
        return 6.0
    if number <= 11:
        return 12.0
    return 25.0


def _neighbor_columns(frame: pd.DataFrame) -> list[tuple[pd.Series, pd.Series]]:
    normalized = {_normal(col): col for col in frame.columns}
    pairs: list[tuple[pd.Series, pd.Series]] = []
    for rank in range(1, 6):
        pci_name = next(
            (
                original
                for norm, original in normalized.items()
                if f"top {rank} pci" in norm or norm == f"neighbor {rank} pci"
            ),
            None,
        )
        rsrp_name = next(
            (
                original
                for norm, original in normalized.items()
                if (f"top {rank}" in norm and "rsrp" in norm)
                or norm == f"neighbor {rank} rsrp dbm"
            ),
            None,
        )
        if pci_name and rsrp_name:
            pairs.append(
                (
                    pd.to_numeric(frame[pci_name].replace("-", np.nan), errors="coerce"),
                    pd.to_numeric(frame[rsrp_name].replace("-", np.nan), errors="coerce"),
                )
            )
    return pairs


def _transition_features(pci: pd.Series) -> tuple[float, float, float]:
    sequence = [int(v) for v in pci if pd.notna(v)]
    if not sequence:
        return math.nan, math.nan, math.nan
    transitions = sum(a != b for a, b in zip(sequence, sequence[1:]))
    ping_pongs = sum(a == c and a != b for a, b, c in zip(sequence, sequence[1:], sequence[2:]))
    rate = transitions / max(len(sequence) - 1, 1)
    return float(transitions), float(rate), float(ping_pongs)


def extract_diagnostic_features(question: str, parsed: ParsedQuestion) -> dict[str, float]:
    """Extract robust scalar evidence from one parsed telecom question."""
    if not parsed.is_telecom or "engineering" not in parsed.tables:
        raise ValueError("Diagnostic features require drive-test and engineering tables")

    drive = parsed.tables["drive_test"].frame.copy()
    engineering = parsed.tables["engineering"].frame.copy()
    threshold = throughput_threshold(question)

    throughput = _numeric(
        drive,
        "5G KPI PCell Layer2 MAC DL Throughput [Mbps]",
        "Throughput(Mbps)",
    )
    speed = _numeric(drive, "GPS Speed (km/h)")
    serving_pci = _numeric(drive, "5G KPI PCell RF Serving PCI", "Serving PCI")
    serving_rsrp = _numeric(
        drive,
        "5G KPI PCell RF Serving SS-RSRP [dBm]",
        "Serving RSRP(dBm)",
    )
    serving_sinr = _numeric(
        drive,
        "5G KPI PCell RF Serving SS-SINR [dB]",
        "Serving SINR(dB)",
    )
    rbs = _numeric(drive, "5G KPI PCell Layer1 DL RB Num (Including 0)", "RB/slot")
    latitude = _numeric(drive, "Latitude")
    longitude = _numeric(drive, "Longitude")

    low = throughput.lt(threshold) & throughput.notna()
    if not low.any() and throughput.notna().any():
        low = throughput.eq(throughput.min()) & throughput.notna()

    transitions, transition_rate, ping_pongs = _transition_features(serving_pci)
    low_transitions, low_transition_rate, low_ping_pongs = _transition_features(serving_pci[low])
    ep_map = _engineering_map(engineering)

    distance = pd.Series(np.nan, index=drive.index, dtype=float)
    total_tilt = pd.Series(np.nan, index=drive.index, dtype=float)
    mechanical_tilt = pd.Series(np.nan, index=drive.index, dtype=float)
    antenna_height = pd.Series(np.nan, index=drive.index, dtype=float)
    tilt_excess = pd.Series(np.nan, index=drive.index, dtype=float)
    serving_gnodeb: dict[int, str | None] = {}
    for index in drive.index:
        if pd.isna(serving_pci[index]):
            continue
        info = ep_map.get(int(serving_pci[index]))
        if not info:
            continue
        serving_gnodeb[index] = info["gnodeb"]  # type: ignore[assignment]
        total_tilt[index] = info["total_tilt"]
        mechanical_tilt[index] = info["mechanical_tilt"]
        antenna_height[index] = info["height"]
        if all(
            pd.notna(value)
            for value in (latitude[index], longitude[index], info["lat"], info["lon"])
        ):
            distance[index] = _haversine_km(
                float(latitude[index]),
                float(longitude[index]),
                float(info["lat"]),
                float(info["lon"]),
            )
            if distance[index] > 0 and pd.notna(info["height"]) and pd.notna(info["total_tilt"]):
                depression = math.degrees(
                    math.atan2(float(info["height"]), float(distance[index]) * 1000)
                )
                tilt_excess[index] = (
                    float(info["total_tilt"])
                    - depression
                    - _vertical_beamwidth(info["beam_scenario"]) / 2
                )

    neighbor_pairs = _neighbor_columns(drive)
    best_neighbor_rsrp = pd.Series(np.nan, index=drive.index, dtype=float)
    mod30_conflict = pd.Series(False, index=drive.index, dtype=bool)
    strong_noncolocated = pd.Series(False, index=drive.index, dtype=bool)
    close_noncolocated = pd.Series(False, index=drive.index, dtype=bool)
    neighbor_observed = pd.Series(False, index=drive.index, dtype=bool)

    serving_throughput_mean = throughput.groupby(serving_pci).mean().to_dict()
    neighbor_gain = pd.Series(np.nan, index=drive.index, dtype=float)
    for neighbor_pci, neighbor_rsrp in neighbor_pairs:
        best_neighbor_rsrp = pd.concat([best_neighbor_rsrp, neighbor_rsrp], axis=1).max(axis=1)
        valid = serving_pci.notna() & neighbor_pci.notna()
        neighbor_observed |= valid
        mod30_conflict |= valid & ((serving_pci % 30) == (neighbor_pci % 30))
        for index in drive.index[valid & throughput.notna()]:
            candidate_mean = serving_throughput_mean.get(neighbor_pci[index])
            if candidate_mean is not None and pd.notna(candidate_mean):
                gain = float(candidate_mean) - float(throughput[index])
                if pd.isna(neighbor_gain[index]) or gain > neighbor_gain[index]:
                    neighbor_gain[index] = gain
        for index in drive.index[valid & neighbor_rsrp.gt(-105)]:
            serving_info = ep_map.get(int(serving_pci[index]))
            neighbor_info = ep_map.get(int(neighbor_pci[index]))
            if serving_info and neighbor_info and serving_info["gnodeb"] != neighbor_info["gnodeb"]:
                strong_noncolocated[index] = True
                if pd.notna(serving_rsrp[index]) and neighbor_rsrp[index] >= serving_rsrp[index] - 3:
                    close_noncolocated[index] = True

    neighbor_margin = best_neighbor_rsrp - serving_rsrp
    neighbor_signal_valid = best_neighbor_rsrp.notna() & serving_rsrp.notna()

    return {
        "row_count": float(len(drive)),
        "throughput_threshold": threshold,
        "low_row_count": float(low.sum()),
        "low_fraction": float(low.mean()),
        "throughput_min": _safe_min(throughput),
        "throughput_mean": _safe_mean(throughput),
        "low_throughput_mean": _safe_mean(throughput[low]),
        "speed_max": _safe_max(speed),
        "low_speed_max": _safe_max(speed[low]),
        "low_speed_mean": _safe_mean(speed[low]),
        "rbs_mean": _safe_mean(rbs),
        "low_rbs_mean": _safe_mean(rbs[low]),
        "low_rbs_below_160_fraction": _fraction(rbs.lt(160) & low, rbs.notna() & low),
        "serving_rsrp_mean": _safe_mean(serving_rsrp),
        "low_serving_rsrp_mean": _safe_mean(serving_rsrp[low]),
        "low_serving_rsrp_min": _safe_min(serving_rsrp[low]),
        "serving_sinr_mean": _safe_mean(serving_sinr),
        "low_serving_sinr_mean": _safe_mean(serving_sinr[low]),
        "serving_cell_count": float(serving_pci.nunique()),
        "handover_count": transitions,
        "handover_rate": transition_rate,
        "ping_pong_count": ping_pongs,
        "low_handover_count": low_transitions,
        "low_handover_rate": low_transition_rate,
        "low_ping_pong_count": low_ping_pongs,
        "distance_max_km": _safe_max(distance),
        "low_distance_max_km": _safe_max(distance[low]),
        "low_distance_mean_km": _safe_mean(distance[low]),
        "serving_total_tilt_max": _safe_max(total_tilt),
        "low_serving_total_tilt_max": _safe_max(total_tilt[low]),
        "low_serving_total_tilt_mean": _safe_mean(total_tilt[low]),
        "low_tilt_excess_max_deg": _safe_max(tilt_excess[low]),
        "low_tilt_excess_mean_deg": _safe_mean(tilt_excess[low]),
        "low_mechanical_tilt_max": _safe_max(mechanical_tilt[low]),
        "low_antenna_height_mean": _safe_mean(antenna_height[low]),
        "neighbor_margin_max_db": _safe_max(neighbor_margin),
        "low_neighbor_margin_max_db": _safe_max(neighbor_margin[low]),
        "low_neighbor_stronger_fraction": _fraction(
            neighbor_margin.gt(0) & low, neighbor_signal_valid & low
        ),
        "low_neighbor_throughput_gain_max": _safe_max(neighbor_gain[low]),
        "low_neighbor_throughput_gain_positive_fraction": _fraction(
            neighbor_gain.gt(0) & low, neighbor_gain.notna() & low
        ),
        "mod30_conflict_fraction": _fraction(mod30_conflict, neighbor_observed),
        "low_mod30_conflict_fraction": _fraction(
            mod30_conflict & low, neighbor_observed & low
        ),
        "strong_noncolocated_fraction": float(strong_noncolocated.mean()),
        "low_strong_noncolocated_fraction": _fraction(strong_noncolocated & low, low),
        "low_close_noncolocated_fraction": _fraction(close_noncolocated & low, low),
        "engineering_match_fraction": float(distance.notna().mean()),
        "parser_markdown": float(parsed.format_name == "markdown"),
    }
