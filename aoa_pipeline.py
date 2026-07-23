from __future__ import annotations

from dataclasses import dataclass, field, replace
from itertools import combinations, product
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from pyubx2 import UBXReader

try:
    from pyubx2 import ERR_IGNORE
except ImportError:  # pragma: no cover - depends on pyubx2 version
    ERR_IGNORE = None


RAWX_COLUMNS = [
    "rcvTow",
    "week",
    "measurement_index",
    "gnssId",
    "svid",
    "sigId",
    "cpMes",
    "prMes",
    "doMes",
    "cno",
    "locktime",
    "cp_valid",
    "pr_valid",
    "half_cycle",
    "cpStd",
]

NAVSAT_COLUMNS = [
    "rcvTow",
    "iTOW",
    "navsat_index",
    "gnssId",
    "svid",
    "cno",
    "elev",
    "azim",
    "qualityInd",
    "svUsed",
]

NAVPVT_COLUMNS = [
    "rcvTow",
    "iTOW",
    "fixType",
    "numSV",
    "lon",
    "lat",
    "height",
    "hMSL",
    "hAcc",
    "vAcc",
    "nano",
]


@dataclass
class DatasetConfig:
    name: str
    receiver1_path: str | Path
    receiver2_path: str | Path
    alpha_r_deg: float | None = None
    reference_svid: int | None = None
    excluded_reference_svids: tuple[int, ...] | None = None
    reference_required_svids: tuple[int, ...] | None = None
    selected_svids: tuple[int, ...] | None = None
    excluded_analysis_svids: tuple[int, ...] | None = None
    start_index: int = 0
    length: int | None = 500
    gnss_id: int = 0
    sig_id: int = 0
    sync_tow_decimals: int | None = None
    use_navpvt_time_correction: bool = True
    navpvt_tow_decimals: int = 0
    baseline_over_wavelength: float = 1.0
    single_difference_order: str = "rx1-rx2"
    branch_values: tuple[int, ...] = (-2, -1, 0, 1, 2)
    navsat_azimuth_bias_deg: float | None = None
    output_root: str | Path = Path("outputs/aoa_paper")

    @property
    def output_dir(self) -> Path:
        return Path(self.output_root) / self.name


@dataclass
class UbxObservationTables:
    rawx: pd.DataFrame
    navsat: pd.DataFrame
    navpvt: pd.DataFrame


@dataclass
class SyncedObservations:
    epochs: pd.DataFrame
    rx1: UbxObservationTables
    rx2: UbxObservationTables


@dataclass
class CarrierPhaseTables:
    rx1_cycles: pd.DataFrame
    rx2_cycles: pd.DataFrame
    combined_cycles: pd.DataFrame
    mean_cno: pd.DataFrame


@dataclass
class AoaRunResult:
    config: DatasetConfig
    epochs: pd.DataFrame
    reference_svid: int
    reference_selection: pd.DataFrame
    carrier_phase_cycles: pd.DataFrame
    carrier_phase_rx1_cycles: pd.DataFrame
    carrier_phase_rx2_cycles: pd.DataFrame
    single_difference_cycles: pd.DataFrame
    double_difference_cycles: pd.DataFrame
    rounded_dd_cycles: pd.DataFrame
    fractional_dd_cycles: pd.DataFrame
    candidate_aoa_deg: pd.DataFrame
    selected_aoa_deg: pd.DataFrame
    selected_branch: pd.DataFrame
    collapse_metrics_by_epoch: pd.DataFrame
    collapse_metrics: pd.DataFrame
    aoa_collapse_metrics_by_epoch: pd.DataFrame
    aoa_collapse_metrics: pd.DataFrame
    navsat_aoa_diagnostic_deg: pd.DataFrame
    ephemeris_aoa_deg: pd.DataFrame
    output_paths: dict[str, Path] = field(default_factory=dict)


@dataclass
class PaperScenarioResult:
    config: DatasetConfig
    partial_spoof_enabled: bool
    spoof_svids: tuple[int, ...]
    probe_result: AoaRunResult | None
    analysis_result: AoaRunResult
    all_satellite_result: AoaRunResult
    reference_alpha_by_epoch: pd.DataFrame
    joint_source_aoa_deg: pd.DataFrame
    joint_source_branch: pd.DataFrame
    pairwise_convergence: pd.DataFrame
    cluster_summary: pd.DataFrame
    all_satellite_dd_method_aoa_deg: pd.DataFrame
    ephemeris_aoa_rx1_deg: pd.DataFrame
    ephemeris_aoa_rx2_deg: pd.DataFrame
    all_satellite_valid_m_candidates: pd.DataFrame
    all_satellite_valid_m_summary: pd.DataFrame
    ephemeris_aligned_dd_method_aoa_deg: pd.DataFrame
    ephemeris_aligned_branch: pd.DataFrame
    best_m_all_satellite_aoa_deg: pd.DataFrame
    aoa_method_equivalence_summary: pd.DataFrame
    sv_used_rx1: pd.DataFrame
    sv_used_rx2: pd.DataFrame
    sv_used_summary: pd.DataFrame
    run_summary: pd.DataFrame
    output_paths: dict[str, Path] = field(default_factory=dict)


def _make_ubx_reader(stream) -> UBXReader:
    kwargs = {"protfilter": 2}
    if ERR_IGNORE is not None:
        kwargs["quitonerror"] = ERR_IGNORE
    return UBXReader(stream, **kwargs)


def _empty_frame(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def _float_or_nan(value: object) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _int_or_none(value: object) -> int | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalise_svids(svids: Sequence[int] | None) -> tuple[int, ...] | None:
    if svids is None:
        return None
    return tuple(dict.fromkeys(int(svid) for svid in svids))


def _rawx_rows(parsed_data, gnss_id: int, sig_id: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    num_meas = _int_or_none(getattr(parsed_data, "numMeas", None))
    if num_meas is None:
        num_meas = 0
        while hasattr(parsed_data, f"gnssId_{num_meas + 1:02}"):
            num_meas += 1

    for index in range(1, num_meas + 1):
        gnss = _int_or_none(getattr(parsed_data, f"gnssId_{index:02}", None))
        sig = _int_or_none(getattr(parsed_data, f"sigId_{index:02}", None))
        if gnss != gnss_id or sig != sig_id:
            continue

        rows.append(
            {
                "rcvTow": _float_or_nan(parsed_data.rcvTow),
                "week": _int_or_none(getattr(parsed_data, "week", None)),
                "measurement_index": index,
                "gnssId": gnss,
                "svid": _int_or_none(getattr(parsed_data, f"svId_{index:02}", None)),
                "sigId": sig,
                "cpMes": _float_or_nan(getattr(parsed_data, f"cpMes_{index:02}", None)),
                "prMes": _float_or_nan(getattr(parsed_data, f"prMes_{index:02}", None)),
                "doMes": _float_or_nan(getattr(parsed_data, f"doMes_{index:02}", None)),
                "cno": _float_or_nan(getattr(parsed_data, f"cno_{index:02}", None)),
                "locktime": _float_or_nan(
                    getattr(parsed_data, f"locktime_{index:02}", None)
                ),
                "cp_valid": _int_or_none(
                    getattr(parsed_data, f"cpValid_{index:02}", None)
                ),
                "pr_valid": _int_or_none(
                    getattr(parsed_data, f"prValid_{index:02}", None)
                ),
                "half_cycle": _int_or_none(
                    getattr(parsed_data, f"halfCyc_{index:02}", None)
                ),
                "cpStd": _float_or_nan(getattr(parsed_data, f"cpStd_{index:02}", None)),
            }
        )
    return rows


def _navsat_rows(parsed_data, gnss_id: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    rcv_tow = _float_or_nan(getattr(parsed_data, "iTOW", float("nan"))) / 1000.0
    num_svs = _int_or_none(getattr(parsed_data, "numSvs", None))
    if num_svs is None:
        num_svs = 0
        while hasattr(parsed_data, f"gnssId_{num_svs + 1:02}"):
            num_svs += 1

    for index in range(1, num_svs + 1):
        gnss = _int_or_none(getattr(parsed_data, f"gnssId_{index:02}", None))
        if gnss != gnss_id:
            continue

        rows.append(
            {
                "rcvTow": rcv_tow,
                "iTOW": _int_or_none(getattr(parsed_data, "iTOW", None)),
                "navsat_index": index,
                "gnssId": gnss,
                "svid": _int_or_none(getattr(parsed_data, f"svId_{index:02}", None)),
                "cno": _float_or_nan(getattr(parsed_data, f"cno_{index:02}", None)),
                "elev": _float_or_nan(getattr(parsed_data, f"elev_{index:02}", None)),
                "azim": _float_or_nan(getattr(parsed_data, f"azim_{index:02}", None)),
                "qualityInd": _int_or_none(
                    getattr(parsed_data, f"qualityInd_{index:02}", None)
                ),
                "svUsed": _int_or_none(
                    getattr(parsed_data, f"svUsed_{index:02}", None)
                ),
            }
        )
    return rows


def _navpvt_row(parsed_data) -> dict[str, object]:
    i_tow = _int_or_none(getattr(parsed_data, "iTOW", None))
    return {
        "rcvTow": (i_tow / 1000.0) if i_tow is not None else float("nan"),
        "iTOW": i_tow,
        "fixType": _int_or_none(getattr(parsed_data, "fixType", None)),
        "numSV": _int_or_none(getattr(parsed_data, "numSV", None)),
        "lon": _float_or_nan(getattr(parsed_data, "lon", None)),
        "lat": _float_or_nan(getattr(parsed_data, "lat", None)),
        "height": _float_or_nan(getattr(parsed_data, "height", None)),
        "hMSL": _float_or_nan(getattr(parsed_data, "hMSL", None)),
        "hAcc": _float_or_nan(getattr(parsed_data, "hAcc", None)),
        "vAcc": _float_or_nan(getattr(parsed_data, "vAcc", None)),
        "nano": _float_or_nan(getattr(parsed_data, "nano", None)),
    }


def read_ubx_observations(
    path: str | Path,
    gnss_id: int = 0,
    sig_id: int = 0,
    max_rawx_epochs: int | None = None,
) -> UbxObservationTables:
    """Parse UBX RXM-RAWX, NAV-SAT, and NAV-PVT records into DataFrames."""
    ubx_path = Path(path)
    if not ubx_path.exists():
        raise FileNotFoundError(ubx_path)

    rawx_rows: list[dict[str, object]] = []
    navsat_rows: list[dict[str, object]] = []
    navpvt_rows: list[dict[str, object]] = []
    rawx_epoch_count = 0

    with ubx_path.open("rb") as stream:
        for _, parsed_data in _make_ubx_reader(stream):
            identity = getattr(parsed_data, "identity", "")
            if identity == "RXM-RAWX":
                rawx_epoch_count += 1
                rawx_rows.extend(
                    _rawx_rows(parsed_data, gnss_id=gnss_id, sig_id=sig_id)
                )
            elif identity == "NAV-SAT":
                navsat_rows.extend(_navsat_rows(parsed_data, gnss_id=gnss_id))
            elif identity == "NAV-PVT":
                navpvt_rows.append(_navpvt_row(parsed_data))
            if max_rawx_epochs is not None and rawx_epoch_count >= max_rawx_epochs:
                break

    rawx = pd.DataFrame(rawx_rows, columns=RAWX_COLUMNS)
    navsat = pd.DataFrame(navsat_rows, columns=NAVSAT_COLUMNS)
    navpvt = pd.DataFrame(navpvt_rows, columns=NAVPVT_COLUMNS)
    return UbxObservationTables(rawx=rawx, navsat=navsat, navpvt=navpvt)


def _with_sync_tow(rawx: pd.DataFrame, tow_round_decimals: int | None) -> pd.DataFrame:
    data = rawx.copy()
    if tow_round_decimals is None:
        data["sync_tow"] = data["rcvTow"]
    else:
        data["sync_tow"] = data["rcvTow"].round(tow_round_decimals)
    return data


def _sync_keys(left: pd.DataFrame, right: pd.DataFrame) -> list[str]:
    if "week" in left.columns and "week" in right.columns:
        if left["week"].notna().any() and right["week"].notna().any():
            return ["week", "sync_tow"]
    return ["sync_tow"]


def sync_receiver_epochs(
    obs1: UbxObservationTables,
    obs2: UbxObservationTables,
    tow_round_decimals: int | None = None,
) -> SyncedObservations:
    """Synchronize two receivers by the intersection of RAWX TOW epoch keys."""
    if obs1.rawx.empty or obs2.rawx.empty:
        raise ValueError("Both receivers must contain RXM-RAWX observations.")

    rx1_rawx_all = _with_sync_tow(obs1.rawx, tow_round_decimals)
    rx2_rawx_all = _with_sync_tow(obs2.rawx, tow_round_decimals)
    keys = _sync_keys(rx1_rawx_all, rx2_rawx_all)
    left_epochs = (
        rx1_rawx_all[keys + ["rcvTow"]]
        .drop_duplicates()
        .rename(columns={"rcvTow": "rcvTow_rx1"})
    )
    right_epochs = (
        rx2_rawx_all[keys + ["rcvTow"]]
        .drop_duplicates()
        .rename(columns={"rcvTow": "rcvTow_rx2"})
    )
    epochs = (
        left_epochs.merge(right_epochs, how="inner", on=keys)
        .sort_values(keys)
        .reset_index(drop=True)
    )
    epochs.insert(0, "epoch_index", np.arange(len(epochs), dtype=int))
    if epochs.empty:
        raise ValueError("No RAWX TOW overlap exists between receivers.")
    epochs["tow_delta_s"] = epochs["rcvTow_rx1"] - epochs["rcvTow_rx2"]

    rx1_rawx = rx1_rawx_all.merge(epochs[keys + ["epoch_index"]], how="inner", on=keys)
    rx2_rawx = rx2_rawx_all.merge(epochs[keys + ["epoch_index"]], how="inner", on=keys)

    return SyncedObservations(
        epochs=epochs,
        rx1=UbxObservationTables(rawx=rx1_rawx, navsat=obs1.navsat, navpvt=obs1.navpvt),
        rx2=UbxObservationTables(rawx=rx2_rawx, navsat=obs2.navsat, navpvt=obs2.navpvt),
    )


def _slice_synced_epochs(
    synced: SyncedObservations,
    start_index: int,
    length: int | None,
) -> SyncedObservations:
    if start_index < 0:
        raise ValueError("start_index must be non-negative.")

    stop = None if length is None else start_index + length
    selected = synced.epochs.iloc[start_index:stop].copy()
    if selected.empty:
        raise ValueError("The requested epoch window is empty.")

    old_to_new = selected[["epoch_index"]].copy()
    old_to_new = old_to_new.rename(columns={"epoch_index": "source_epoch_index"})
    old_to_new["epoch_index"] = np.arange(len(old_to_new), dtype=int)

    selected = selected.rename(columns={"epoch_index": "source_epoch_index"})
    selected = selected.merge(old_to_new, how="inner", on="source_epoch_index")
    leading_cols = ["epoch_index", "source_epoch_index"]
    other_cols = [col for col in selected.columns if col not in leading_cols]
    selected = selected[leading_cols + other_cols]

    def remap(rawx: pd.DataFrame) -> pd.DataFrame:
        mapped = rawx.rename(columns={"epoch_index": "source_epoch_index"})
        mapped = mapped.merge(old_to_new, how="inner", on="source_epoch_index")
        leading = ["epoch_index", "source_epoch_index"]
        rest = [col for col in mapped.columns if col not in leading]
        return mapped[leading + rest]

    return SyncedObservations(
        epochs=selected.reset_index(drop=True),
        rx1=UbxObservationTables(
            rawx=remap(synced.rx1.rawx),
            navsat=synced.rx1.navsat,
            navpvt=synced.rx1.navpvt,
        ),
        rx2=UbxObservationTables(
            rawx=remap(synced.rx2.rawx),
            navsat=synced.rx2.navsat,
            navpvt=synced.rx2.navpvt,
        ),
    )


def _pivot_rawx(
    rawx: pd.DataFrame,
    value_col: str,
    epoch_index: pd.Index,
    require_valid_phase: bool = False,
) -> pd.DataFrame:
    if rawx.empty:
        return pd.DataFrame(index=epoch_index)

    valid = rawx[value_col].notna() & rawx["svid"].notna()
    if require_valid_phase and rawx["cp_valid"].notna().any():
        valid &= rawx["cp_valid"].fillna(0).astype(int).eq(1)

    pivot = rawx.loc[valid].pivot_table(
        index="epoch_index",
        columns="svid",
        values=value_col,
        aggfunc="first",
    )
    pivot = pivot.reindex(epoch_index)
    pivot.index.name = "epoch_index"
    pivot.columns = [int(col) for col in pivot.columns]
    return pivot.sort_index(axis=1)


def _attach_navpvt_time_correction(
    rawx: pd.DataFrame,
    navpvt: pd.DataFrame,
    tow_round_decimals: int,
    use_navpvt_time_correction: bool,
) -> pd.DataFrame:
    data = rawx.copy()
    data["carrier_phase_correction_cycles"] = 0.0
    data["navpvt_nano"] = np.nan
    if not use_navpvt_time_correction:
        data["carrier_phase_cycles"] = data["cpMes"]
        return data

    if navpvt.empty:
        data["carrier_phase_correction_cycles"] = np.nan
        data["carrier_phase_cycles"] = np.nan
        return data

    data["_navpvt_sync_tow"] = data["rcvTow"].round(tow_round_decimals)
    nano = navpvt[["rcvTow", "nano"]].dropna(subset=["rcvTow", "nano"]).copy()
    nano["_navpvt_sync_tow"] = nano["rcvTow"].round(tow_round_decimals)
    nano = (
        nano.sort_values("rcvTow")
        .drop_duplicates("_navpvt_sync_tow", keep="first")
        .loc[:, ["_navpvt_sync_tow", "nano"]]
        .rename(columns={"nano": "navpvt_nano"})
    )

    data = data.merge(
        nano, how="left", on="_navpvt_sync_tow", suffixes=("", "_matched")
    )
    data["navpvt_nano"] = data["navpvt_nano_matched"].combine_first(data["navpvt_nano"])
    data = data.drop(columns=["navpvt_nano_matched", "_navpvt_sync_tow"])
    data["carrier_phase_correction_cycles"] = data["navpvt_nano"] * 1e-9 * data["doMes"]
    data["carrier_phase_cycles"] = (
        data["cpMes"] + data["carrier_phase_correction_cycles"]
    )
    return data


def _filter_carrier_phase_columns(
    tables: CarrierPhaseTables,
    svids: Sequence[int] | None,
) -> CarrierPhaseTables:
    if svids is None:
        return tables
    columns = [svid for svid in _normalise_svids(svids) or ()]
    return CarrierPhaseTables(
        rx1_cycles=tables.rx1_cycles.reindex(columns=columns),
        rx2_cycles=tables.rx2_cycles.reindex(columns=columns),
        combined_cycles=tables.combined_cycles.reindex(
            columns=pd.MultiIndex.from_product([["rx1", "rx2"], columns])
        ),
        mean_cno=tables.mean_cno.reindex(columns=columns),
    )


def build_carrier_phase_table(
    synced_epochs: SyncedObservations,
    use_navpvt_time_correction: bool = True,
    navpvt_tow_decimals: int = 0,
) -> CarrierPhaseTables:
    """Build RX1/RX2 GPS L1 carrier-phase tables indexed by synchronized epoch."""
    epoch_index = pd.Index(synced_epochs.epochs["epoch_index"], name="epoch_index")
    rx1_rawx = _attach_navpvt_time_correction(
        synced_epochs.rx1.rawx,
        synced_epochs.rx1.navpvt,
        navpvt_tow_decimals,
        use_navpvt_time_correction,
    )
    rx2_rawx = _attach_navpvt_time_correction(
        synced_epochs.rx2.rawx,
        synced_epochs.rx2.navpvt,
        navpvt_tow_decimals,
        use_navpvt_time_correction,
    )
    rx1 = _pivot_rawx(rx1_rawx, "carrier_phase_cycles", epoch_index, True)
    rx2 = _pivot_rawx(rx2_rawx, "carrier_phase_cycles", epoch_index, True)
    cno1 = _pivot_rawx(rx1_rawx, "cno", epoch_index)
    cno2 = _pivot_rawx(rx2_rawx, "cno", epoch_index)

    columns = sorted(set(rx1.columns) | set(rx2.columns))
    rx1 = rx1.reindex(columns=columns)
    rx2 = rx2.reindex(columns=columns)
    cno1 = cno1.reindex(columns=columns)
    cno2 = cno2.reindex(columns=columns)

    combined = pd.concat({"rx1": rx1, "rx2": rx2}, axis=1)
    mean_cno = pd.concat([cno1, cno2]).groupby(level=0).mean().reindex(epoch_index)
    mean_cno = mean_cno.reindex(columns=columns)
    return CarrierPhaseTables(
        rx1_cycles=rx1,
        rx2_cycles=rx2,
        combined_cycles=combined,
        mean_cno=mean_cno,
    )


def _longest_true_run(values: Iterable[bool]) -> int:
    longest = 0
    current = 0
    for value in values:
        if bool(value):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def select_reference_svid(
    carrier_phase: CarrierPhaseTables,
    cno: pd.DataFrame | None = None,
    override: int | None = None,
    exclude_svids: Sequence[int] | None = None,
    required_svids: Sequence[int] | None = None,
) -> tuple[int, pd.DataFrame]:
    """Select a reference satellite, honoring an explicit override first.

    ``override`` is authoritative: it may name an SVID in ``exclude_svids``. This
    lets experiments deliberately use any observed PRN (including a known spoof
    PRN) as the DD reference. Exclusions only constrain automatic selection.
    """
    rx1 = carrier_phase.rx1_cycles
    rx2 = carrier_phase.rx2_cycles
    common_valid = rx1.notna() & rx2.notna()
    excluded = set(_normalise_svids(exclude_svids) or ())
    required = _normalise_svids(required_svids) or ()

    required_mask = pd.Series(True, index=common_valid.index)
    missing_required = [svid for svid in required if svid not in common_valid.columns]
    if missing_required:
        raise ValueError(
            f"required_svids are not present in both receivers: {missing_required}"
        )
    for svid in required:
        required_mask &= common_valid[svid]

    if override is not None:
        if override not in common_valid.columns:
            raise ValueError(
                f"reference_svid={override} is not present in both receivers."
            )
        if not (common_valid[override] & required_mask).any():
            raise ValueError(
                f"reference_svid={override} has no common valid epochs with required_svids."
            )
        selected = int(override)
    else:
        rows: list[dict[str, object]] = []
        single_diff = rx1 - rx2
        for svid in common_valid.columns:
            if int(svid) in excluded:
                continue
            mask = common_valid[svid] & required_mask
            valid_count = int(mask.sum())
            if valid_count == 0:
                continue
            phase_step = single_diff.loc[mask, svid].diff().abs()
            jitter = float(phase_step.median(skipna=True))
            if np.isnan(jitter):
                jitter = float("inf")
            cno_mean = float("nan")
            if cno is not None and svid in cno.columns:
                cno_mean = float(cno.loc[mask, svid].mean(skipna=True))
            rows.append(
                {
                    "svid": int(svid),
                    "valid_epoch_count": valid_count,
                    "longest_continuous_run": _longest_true_run(mask.to_numpy()),
                    "mean_cno": cno_mean,
                    "median_abs_phase_step_cycles": jitter,
                }
            )

        if not rows:
            raise ValueError(
                "No satellite has common valid carrier phase in both receivers."
            )

        ranking = pd.DataFrame(rows)
        ranking["_mean_cno_sort"] = ranking["mean_cno"].fillna(-np.inf)
        ranking["_jitter_sort"] = ranking["median_abs_phase_step_cycles"].replace(
            np.inf, np.nan
        )
        ranking["_jitter_sort"] = ranking["_jitter_sort"].fillna(np.inf)
        ranking = ranking.sort_values(
            by=[
                "longest_continuous_run",
                "valid_epoch_count",
                "_mean_cno_sort",
                "_jitter_sort",
                "svid",
            ],
            ascending=[False, False, False, True, True],
        ).reset_index(drop=True)
        selected = int(ranking.loc[0, "svid"])

    selection = pd.DataFrame()
    if override is not None:
        selected_mask = common_valid[selected] & required_mask
        selection = pd.DataFrame(
            [
                {
                    "svid": selected,
                    "valid_epoch_count": int(selected_mask.sum()),
                    "longest_continuous_run": _longest_true_run(
                        selected_mask.to_numpy()
                    ),
                    "mean_cno": (
                        float(
                            cno.loc[selected_mask, selected].mean(skipna=True)
                        )
                        if cno is not None and selected in cno.columns
                        else float("nan")
                    ),
                    "median_abs_phase_step_cycles": float("nan"),
                    "manual_override": True,
                }
            ]
        )
    else:
        selection = ranking.drop(columns=["_mean_cno_sort", "_jitter_sort"])
        selection["manual_override"] = False
    return selected, selection


def compute_single_difference(
    carrier_phase_rx1: CarrierPhaseTables | pd.DataFrame,
    carrier_phase_rx2: pd.DataFrame | None = None,
    order: str = "rx1-rx2",
) -> pd.DataFrame:
    """Compute receiver single differences in carrier cycles.

    ``order="rx1-rx2"`` (default) uses ``RX1 - RX2``; ``order="rx2-rx1"`` uses ``RX2 - RX1``,
    which flips the baseline polarity, i.e. negates every downstream double difference and
    DD-method AoA. The order must match the baseline direction assumed by the ephemeris AoA
    (the azimuth bias): the two orders yield supplementary AoA (``alpha`` vs ``180 - alpha``).
    """
    if order not in ("rx1-rx2", "rx2-rx1"):
        raise ValueError(f"order must be 'rx1-rx2' or 'rx2-rx1', got {order!r}")

    if isinstance(carrier_phase_rx1, CarrierPhaseTables):
        rx1 = carrier_phase_rx1.rx1_cycles
        rx2 = carrier_phase_rx1.rx2_cycles
    else:
        if carrier_phase_rx2 is None:
            raise ValueError("carrier_phase_rx2 is required when passing DataFrames.")
        rx1 = carrier_phase_rx1
        rx2 = carrier_phase_rx2

    columns = sorted(set(rx1.columns) | set(rx2.columns))
    result = rx1.reindex(columns=columns) - rx2.reindex(columns=columns)
    if order == "rx2-rx1":
        result = -result
    result.index.name = "epoch_index"
    return result


def compute_double_difference(
    single_difference: pd.DataFrame,
    reference_svid: int,
) -> pd.DataFrame:
    """Compute satellite double differences in carrier cycles using a reference SV."""
    if reference_svid not in single_difference.columns:
        raise ValueError(
            f"Reference SVID {reference_svid} is not in single differences."
        )
    result = single_difference.subtract(single_difference[reference_svid], axis=0)
    result.index.name = "epoch_index"
    return result


def fractional_round_dd(
    double_difference: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply Eq. 18: fractional DD = DD cycles - round(DD cycles)."""
    rounded = double_difference.round()
    fractional = double_difference - rounded
    rounded.index.name = "epoch_index"
    fractional.index.name = "epoch_index"
    return rounded, fractional


def generate_aoa_candidates(
    fractional_dd: pd.DataFrame,
    alpha_r_deg: float | pd.Series,
    branch_values: Iterable[int] = (-2, -1, 0, 1, 2),
    baseline_over_wavelength: float = 1.0,
) -> pd.DataFrame:
    """Generate Eq. 21 AoA branch candidates from fractional DD values.

    ``alpha_r_deg`` may be a scalar anchor or an epoch-indexed ephemeris AoA series.
    The latter implements the journal method's time-varying reference direction.
    """
    if baseline_over_wavelength == 0:
        raise ValueError("baseline_over_wavelength must be non-zero.")

    frac = fractional_dd.copy()
    frac.index.name = "epoch_index"
    long = frac.reset_index().melt(
        id_vars=["epoch_index"],
        var_name="svid",
        value_name="fractional_dd_cycles",
    )
    long["svid"] = long["svid"].astype(int)
    if isinstance(alpha_r_deg, pd.Series):
        reference_alpha = pd.to_numeric(alpha_r_deg, errors="coerce").reindex(frac.index)
        reference_alpha.index.name = "epoch_index"
        long["reference_alpha_deg"] = long["epoch_index"].map(reference_alpha)
    else:
        long["reference_alpha_deg"] = float(alpha_r_deg)
    long["reference_cos_alpha"] = np.cos(
        np.deg2rad(long["reference_alpha_deg"].astype(float))
    )

    frames: list[pd.DataFrame] = []
    for branch in branch_values:
        candidate = long.copy()
        candidate["branch"] = int(branch)
        candidate["candidate_cos_alpha"] = (
            candidate["fractional_dd_cycles"] + int(branch)
        ) / baseline_over_wavelength + candidate["reference_cos_alpha"]
        valid = candidate["fractional_dd_cycles"].notna() & candidate[
            "candidate_cos_alpha"
        ].between(-1.0, 1.0, inclusive="both")
        clipped = candidate["candidate_cos_alpha"].clip(-1.0, 1.0)
        candidate["candidate_aoa_deg"] = np.where(
            valid,
            np.rad2deg(np.arccos(clipped)),
            np.nan,
        )
        candidate["valid_candidate"] = valid
        frames.append(candidate)

    return pd.concat(frames, ignore_index=True).drop(columns="reference_cos_alpha")


def _valid_segments(index_values: list[int], valid_mask: pd.Series) -> list[list[int]]:
    segments: list[list[int]] = []
    current: list[int] = []
    previous: int | None = None
    for epoch in index_values:
        valid = bool(valid_mask.loc[epoch])
        if not valid:
            if current:
                segments.append(current)
                current = []
            previous = None
            continue
        if previous is None or epoch == previous + 1:
            current.append(epoch)
        else:
            if current:
                segments.append(current)
            current = [epoch]
        previous = epoch
    if current:
        segments.append(current)
    return segments


def _dp_select_segment(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = values.shape
    costs = np.full((rows, cols), np.inf, dtype=float)
    back = np.full((rows, cols), -1, dtype=int)
    valid_first = np.isfinite(values[0])
    costs[0, valid_first] = 0.0

    for row in range(1, rows):
        for current_col in range(cols):
            current_value = values[row, current_col]
            if not np.isfinite(current_value):
                continue
            previous_values = values[row - 1]
            transition = costs[row - 1] + np.abs(current_value - previous_values)
            best_previous = int(np.nanargmin(transition))
            if np.isfinite(transition[best_previous]):
                costs[row, current_col] = transition[best_previous]
                back[row, current_col] = best_previous

    end_col = int(np.nanargmin(costs[-1]))
    if not np.isfinite(costs[-1, end_col]):
        return np.full(rows, np.nan), np.full(rows, np.nan)

    selected_cols = np.full(rows, end_col, dtype=int)
    for row in range(rows - 1, 0, -1):
        selected_cols[row - 1] = back[row, selected_cols[row]]

    selected_values = values[np.arange(rows), selected_cols]
    return selected_values, selected_cols.astype(float)


def select_aoa_dp(
    candidates: pd.DataFrame,
    reference_svid: int | None = None,
    reference_angle_deg: float | pd.Series | None = None,
    return_branches: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Select a smooth AoA sequence per satellite using Algorithm 1."""
    required = {"epoch_index", "svid", "branch", "candidate_aoa_deg"}
    missing = required - set(candidates.columns)
    if missing:
        raise ValueError(f"candidates is missing required columns: {sorted(missing)}")

    all_epochs = sorted(int(epoch) for epoch in candidates["epoch_index"].unique())
    all_svids = sorted(int(svid) for svid in candidates["svid"].unique())
    branch_values = sorted(int(branch) for branch in candidates["branch"].unique())
    selected = pd.DataFrame(index=all_epochs, columns=all_svids, dtype=float)
    branches = pd.DataFrame(index=all_epochs, columns=all_svids, dtype=float)
    selected.index.name = "epoch_index"
    branches.index.name = "epoch_index"

    for svid in all_svids:
        subset = candidates.loc[candidates["svid"].eq(svid)]
        pivot = subset.pivot_table(
            index="epoch_index",
            columns="branch",
            values="candidate_aoa_deg",
            aggfunc="first",
        )
        pivot = pivot.reindex(index=all_epochs, columns=branch_values)
        valid_mask = pivot.notna().any(axis=1)
        for segment in _valid_segments(all_epochs, valid_mask):
            values = pivot.loc[segment].to_numpy(dtype=float)
            segment_values, selected_cols = _dp_select_segment(values)
            selected.loc[segment, svid] = segment_values
            for epoch, col in zip(segment, selected_cols):
                if np.isfinite(col):
                    branches.loc[epoch, svid] = branch_values[int(col)]

    if reference_svid is not None and reference_svid in selected.columns:
        if reference_angle_deg is not None:
            ref_mask = candidates.loc[
                candidates["svid"].eq(reference_svid)
                & candidates["fractional_dd_cycles"].notna()
            ]
            valid_ref_epochs = sorted(
                int(epoch) for epoch in ref_mask["epoch_index"].unique()
            )
            if isinstance(reference_angle_deg, pd.Series):
                reference_values = pd.to_numeric(
                    reference_angle_deg, errors="coerce"
                ).reindex(valid_ref_epochs)
                selected.loc[valid_ref_epochs, reference_svid] = reference_values.to_numpy()
            else:
                selected.loc[valid_ref_epochs, reference_svid] = float(reference_angle_deg)
            branches.loc[valid_ref_epochs, reference_svid] = 0.0

    if return_branches:
        return selected, branches
    return selected


def candidate_aoa_summary(candidates: pd.DataFrame) -> pd.DataFrame:
    """Per-satellite table of valid branch candidates -- the 'maybe' AoA values.

    For each satellite and branch ``m``, report the median candidate AoA over the epochs where
    that branch is valid (``cos(alpha)`` within ``[-1, 1]``) and the count of such epochs.
    Because ``cos(alpha)`` is bounded to ``[-1, 1]``, extreme branches (``m = +-2``) are usually
    rare or impossible, so most satellites show only one or two candidate angles. Pick the
    correct branch afterwards (e.g. with :func:`select_aoa_by_reference`).
    """
    required = {"svid", "branch", "candidate_aoa_deg"}
    missing = required - set(candidates.columns)
    if missing:
        raise ValueError(f"candidates is missing required columns: {sorted(missing)}")

    valid = candidates.dropna(subset=["candidate_aoa_deg"]).copy()
    if valid.empty:
        return pd.DataFrame()
    valid["svid"] = valid["svid"].astype(int)
    valid["branch"] = valid["branch"].astype(int)

    aoa = valid.pivot_table(
        index="svid", columns="branch", values="candidate_aoa_deg", aggfunc="median"
    ).rename(columns=lambda m: f"aoa_m{int(m):+d}")
    count = valid.pivot_table(
        index="svid", columns="branch", values="candidate_aoa_deg", aggfunc="count"
    ).rename(columns=lambda m: f"n_m{int(m):+d}")
    summary = aoa.join(count)
    summary.index.name = "svid"
    return summary.sort_index()


def select_aoa_by_reference(
    candidates: pd.DataFrame,
    reference_aoa: pd.DataFrame,
    return_branches: bool = False,
    min_valid_fraction: float = 0.8,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.Series]:
    """Resolve one branch per satellite by matching a reference AoA, then return per-epoch AoA.

    For each satellite, pick one branch ``m`` that is valid across most of the satellite's
    valid epochs and has the smallest median error to ``reference_aoa`` (e.g.
    ephemeris-based AoA), then return that branch's candidate AoA at every epoch. This
    resolves the branch ambiguity from geometry instead of from time-smoothness, so the
    absolute level is anchored per satellite.

    Note: matching the branch to the ephemeris biases spoofed PRNs toward agreement, so the
    returned AoA is for authentic-vs-ephemeris comparison, not a standalone spoof test (use the
    fractional-DD residual against a known baseline for detection).
    """
    if not 0 < min_valid_fraction <= 1:
        raise ValueError("min_valid_fraction must be in the interval (0, 1].")

    valid = candidates.dropna(subset=["candidate_aoa_deg"]).copy()
    valid["svid"] = valid["svid"].astype(int)
    valid["branch"] = valid["branch"].astype(int)

    all_epochs = sorted(int(epoch) for epoch in candidates["epoch_index"].unique())
    svids = sorted(int(svid) for svid in valid["svid"].unique())
    reference_median = reference_aoa.median()

    selected = pd.DataFrame(index=all_epochs, columns=svids, dtype=float)
    selected.index.name = "epoch_index"
    chosen: dict[int, int] = {}
    for svid in svids:
        if svid not in reference_median.index or not np.isfinite(reference_median[svid]):
            continue
        subset = valid.loc[valid["svid"].eq(svid)]
        all_valid_epochs = int(subset["epoch_index"].nunique())
        min_epoch_count = max(1, int(np.ceil(all_valid_epochs * min_valid_fraction)))
        branch_epoch_counts = subset.groupby("branch")["epoch_index"].nunique()
        eligible_branches = branch_epoch_counts.loc[
            branch_epoch_counts.ge(min_epoch_count)
        ].index
        if len(eligible_branches) == 0:
            eligible_branches = branch_epoch_counts.loc[
                branch_epoch_counts.eq(branch_epoch_counts.max())
            ].index

        branch_scores: dict[int, float] = {}
        reference = reference_aoa[svid].rename("reference_aoa_deg")
        for branch in eligible_branches:
            branch_subset = subset.loc[subset["branch"].eq(int(branch))]
            comparison = branch_subset.merge(
                reference.reset_index(), on="epoch_index", how="left"
            ).dropna(subset=["reference_aoa_deg"])
            if comparison.empty:
                continue
            branch_scores[int(branch)] = float(
                (
                    comparison["candidate_aoa_deg"]
                    - comparison["reference_aoa_deg"]
                )
                .abs()
                .median()
            )
        if not branch_scores:
            continue

        best_branch = min(branch_scores, key=branch_scores.get)
        chosen[svid] = best_branch
        series = subset.loc[subset["branch"].eq(best_branch)].set_index("epoch_index")[
            "candidate_aoa_deg"
        ]
        selected[svid] = series.reindex(all_epochs)

    if return_branches:
        return selected, pd.Series(chosen, name="chosen_branch").sort_index()
    return selected


def valid_m_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    """Return the physically valid AoA candidates for each SVID/epoch/branch."""
    required = {
        "epoch_index",
        "svid",
        "branch",
        "fractional_dd_cycles",
        "candidate_cos_alpha",
        "candidate_aoa_deg",
        "valid_candidate",
    }
    missing = required - set(candidates.columns)
    if missing:
        raise ValueError(f"candidates is missing required columns: {sorted(missing)}")

    valid = candidates.loc[candidates["valid_candidate"]].copy()
    valid = valid.dropna(subset=["candidate_aoa_deg"])
    if valid.empty:
        return valid
    valid["epoch_index"] = valid["epoch_index"].astype(int)
    valid["svid"] = valid["svid"].astype(int)
    valid["branch"] = valid["branch"].astype(int)
    return valid.sort_values(["svid", "branch", "epoch_index"]).reset_index(drop=True)


def summarize_valid_m_candidates(
    candidates: pd.DataFrame,
    *,
    reference_aoa: pd.DataFrame | None = None,
    selected_branch: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Summarize every physically valid integer branch ``m`` per SVID."""
    valid = valid_m_candidates(candidates)
    if valid.empty:
        return pd.DataFrame()

    rows: list[dict[str, float | int | None]] = []
    for (svid, branch), subset in valid.groupby(["svid", "branch"], sort=True):
        row: dict[str, float | int | None] = {
            "svid": int(svid),
            "branch": int(branch),
            "valid_epoch_count": int(len(subset)),
            "median_candidate_aoa_deg": float(subset["candidate_aoa_deg"].median()),
            "mean_candidate_aoa_deg": float(subset["candidate_aoa_deg"].mean()),
            "min_candidate_aoa_deg": float(subset["candidate_aoa_deg"].min()),
            "max_candidate_aoa_deg": float(subset["candidate_aoa_deg"].max()),
            "median_candidate_cos_alpha": float(
                subset["candidate_cos_alpha"].median()
            ),
        }

        if reference_aoa is not None and int(svid) in reference_aoa.columns:
            reference = reference_aoa[int(svid)].rename("reference_aoa_deg")
            comparison = subset.merge(
                reference.reset_index(), on="epoch_index", how="left"
            ).dropna(subset=["reference_aoa_deg"])
            if not comparison.empty:
                abs_error = (
                    comparison["candidate_aoa_deg"]
                    - comparison["reference_aoa_deg"]
                ).abs()
                row.update(
                    {
                        "reference_epoch_count": int(len(comparison)),
                        "median_reference_aoa_deg": float(
                            comparison["reference_aoa_deg"].median()
                        ),
                        "median_abs_error_to_reference_deg": float(
                            abs_error.median()
                        ),
                        "mean_abs_error_to_reference_deg": float(abs_error.mean()),
                        "within_10deg_fraction": float((abs_error <= 10.0).mean()),
                    }
                )

        if selected_branch is not None and int(svid) in selected_branch.columns:
            branch_series = selected_branch[int(svid)].reindex(subset["epoch_index"])
            row["dp_selected_epoch_fraction"] = float(
                branch_series.eq(int(branch)).mean()
            )

        rows.append(row)

    return pd.DataFrame(rows).sort_values(["svid", "branch"]).reset_index(drop=True)


def _branch_series_to_frame(branches: pd.Series) -> pd.DataFrame:
    """Convert a per-SVID branch Series into a stable CSV-friendly table."""
    if branches.empty:
        return pd.DataFrame(columns=["svid", "branch"])
    frame = branches.rename_axis("svid").reset_index()
    frame.columns = ["svid", "branch"]
    frame["svid"] = frame["svid"].astype(int)
    frame["branch"] = frame["branch"].astype(int)
    return frame.sort_values("svid").reset_index(drop=True)


def summarize_aoa_method_equivalence(
    *,
    dp_aoa: pd.DataFrame,
    dp_branch: pd.DataFrame | None = None,
    ephemeris_aligned_aoa: pd.DataFrame,
    ephemeris_aoa: pd.DataFrame,
    ephemeris_aligned_branch: pd.Series,
) -> pd.DataFrame:
    """Compare DD-method AoA variants against ephemeris-based AoA per SVID."""

    dp_branch_by_svid = pd.Series(dtype=float)
    if dp_branch is not None:
        branch_modes: dict[int, float] = {}
        for column in dp_branch.columns:
            series = dp_branch[column].dropna()
            if not series.empty:
                branch_modes[int(column)] = float(series.mode().iloc[0])
        dp_branch_by_svid = pd.Series(branch_modes)

    def summarize_one(
        method: str,
        dd_aoa: pd.DataFrame,
        branch_by_svid: pd.Series | None = None,
    ) -> list[dict[str, float | int | str | None]]:
        rows: list[dict[str, float | int | str | None]] = []
        for svid in sorted(int(column) for column in dd_aoa.columns):
            if svid not in ephemeris_aoa.columns:
                continue
            joined = pd.concat(
                [
                    dd_aoa[svid].rename("dd_aoa_deg"),
                    ephemeris_aoa[svid].rename("ephemeris_aoa_deg"),
                ],
                axis=1,
            ).dropna()
            if joined.empty:
                continue
            abs_error = (
                joined["dd_aoa_deg"] - joined["ephemeris_aoa_deg"]
            ).abs()
            branch_value: int | None = None
            if branch_by_svid is not None and svid in branch_by_svid.index:
                branch_value = int(branch_by_svid.loc[svid])
            rows.append(
                {
                    "method": method,
                    "svid": int(svid),
                    "branch": branch_value,
                    "epoch_count": int(len(joined)),
                    "median_abs_error_deg": float(abs_error.median()),
                    "mean_abs_error_deg": float(abs_error.mean()),
                    "p90_abs_error_deg": float(abs_error.quantile(0.9)),
                    "within_5deg_fraction": float((abs_error <= 5.0).mean()),
                    "within_10deg_fraction": float((abs_error <= 10.0).mean()),
                }
            )
        return rows

    rows = summarize_one("dp_smoothness", dp_aoa, dp_branch_by_svid)
    rows.extend(
        summarize_one(
            "ephemeris_aligned_constant_m",
            ephemeris_aligned_aoa,
            ephemeris_aligned_branch,
        )
    )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["svid", "method"]).reset_index(drop=True)


def build_best_m_all_satellite_aoa(
    dp_aoa: pd.DataFrame,
    ephemeris_aligned_aoa: pd.DataFrame,
    spoof_svids: Sequence[int] = (),
) -> pd.DataFrame:
    """Use the best (ephemeris-aligned) branch ``m`` for authentic SVs, DP branch for spoofed SVs.

    The time-smoothness DP (:func:`select_aoa_dp`) sometimes locks onto the wrong integer branch
    ``m`` for an authentic satellite, which inflates its DD-vs-ephemeris error even though the
    signal is genuine (e.g. SV24/SV25 in the partial-spoof set). For authentic SVs this swaps in
    the ephemeris-aligned constant-m branch from :func:`select_aoa_by_reference` -- the "best m"
    visible in the brute-force candidate plot -- which resolves the branch from geometry.

    Spoofed SVs deliberately keep the DP branch: ephemeris-aligning a spoofed PRN biases it
    toward the very ephemeris under test and scatters it off the common-source collapse, so the
    spoof signature is preserved by leaving those columns untouched. Authentic SVs without an
    ephemeris reference (no NAV-SAT AoA) also fall back to the DP branch.
    """
    spoof = set(_normalise_svids(spoof_svids) or ())
    ephemeris_by_svid = {
        int(column): column for column in ephemeris_aligned_aoa.columns
    }
    combined = dp_aoa.copy()
    for column in combined.columns:
        svid = int(column)
        if svid in spoof or svid not in ephemeris_by_svid:
            continue
        combined[column] = ephemeris_aligned_aoa[ephemeris_by_svid[svid]].reindex(
            combined.index
        )
    if combined.empty:
        return combined
    return sort_svid_columns(combined)


def compute_collapse_metrics(
    selected_aoa: pd.DataFrame,
    thresholds_deg: Iterable[float] = (5.0, 10.0, 15.0),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute cross-satellite AoA dispersion metrics for a DD-only run."""
    valid_count = selected_aoa.count(axis=1)
    epoch_median = selected_aoa.median(axis=1, skipna=True)
    epoch_min = selected_aoa.min(axis=1, skipna=True)
    epoch_max = selected_aoa.max(axis=1, skipna=True)
    dispersion = epoch_max - epoch_min
    mad = selected_aoa.subtract(epoch_median, axis=0).abs().median(axis=1, skipna=True)

    per_epoch = pd.DataFrame(
        {
            "valid_satellite_count": valid_count,
            "median_aoa_deg": epoch_median,
            "min_aoa_deg": epoch_min,
            "max_aoa_deg": epoch_max,
            "max_minus_min_dispersion_deg": dispersion,
            "median_absolute_deviation_deg": mad,
        }
    )
    per_epoch.index.name = "epoch_index"

    usable = per_epoch.loc[per_epoch["valid_satellite_count"].ge(2)]
    row: dict[str, object] = {
        "usable_epoch_count": int(len(usable)),
        "total_epoch_count": int(len(per_epoch)),
        "median_dispersion_deg": float(
            usable["max_minus_min_dispersion_deg"].median(skipna=True)
        )
        if not usable.empty
        else float("nan"),
        "p90_dispersion_deg": float(
            usable["max_minus_min_dispersion_deg"].quantile(0.9)
        )
        if not usable.empty
        else float("nan"),
        "median_mad_deg": float(
            usable["median_absolute_deviation_deg"].median(skipna=True)
        )
        if not usable.empty
        else float("nan"),
    }
    for threshold in thresholds_deg:
        key = f"pct_epochs_dispersion_le_{int(threshold)}deg"
        row[key] = (
            float((usable["max_minus_min_dispersion_deg"] <= threshold).mean() * 100.0)
            if not usable.empty
            else float("nan")
        )

    summary = pd.DataFrame([row])
    return per_epoch, summary


def compute_relative_dd_collapse_metrics(
    fractional_dd: pd.DataFrame,
    thresholds_cycles: Iterable[float] = (0.05, 0.1, 0.15),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute cross-satellite collapse from fractional DD values in cycles."""
    valid_count = fractional_dd.count(axis=1)
    epoch_median = fractional_dd.median(axis=1, skipna=True)
    epoch_min = fractional_dd.min(axis=1, skipna=True)
    epoch_max = fractional_dd.max(axis=1, skipna=True)
    dispersion = epoch_max - epoch_min
    mad = fractional_dd.subtract(epoch_median, axis=0).abs().median(axis=1, skipna=True)

    per_epoch = pd.DataFrame(
        {
            "valid_satellite_count": valid_count,
            "median_fractional_dd_cycles": epoch_median,
            "min_fractional_dd_cycles": epoch_min,
            "max_fractional_dd_cycles": epoch_max,
            "max_minus_min_dispersion_cycles": dispersion,
            "median_absolute_deviation_cycles": mad,
        }
    )
    per_epoch.index.name = "epoch_index"

    usable = per_epoch.loc[per_epoch["valid_satellite_count"].ge(2)]
    row: dict[str, object] = {
        "usable_epoch_count": int(len(usable)),
        "total_epoch_count": int(len(per_epoch)),
        "median_dispersion_cycles": float(
            usable["max_minus_min_dispersion_cycles"].median(skipna=True)
        )
        if not usable.empty
        else float("nan"),
        "p90_dispersion_cycles": float(
            usable["max_minus_min_dispersion_cycles"].quantile(0.9)
        )
        if not usable.empty
        else float("nan"),
        "median_mad_cycles": float(
            usable["median_absolute_deviation_cycles"].median(skipna=True)
        )
        if not usable.empty
        else float("nan"),
    }
    for threshold in thresholds_cycles:
        threshold_key = str(threshold).replace(".", "p")
        key = f"pct_epochs_dispersion_le_{threshold_key}_cycles"
        row[key] = (
            float(
                (usable["max_minus_min_dispersion_cycles"] <= threshold).mean() * 100.0
            )
            if not usable.empty
            else float("nan")
        )

    summary = pd.DataFrame([row])
    return per_epoch, summary


def _drop_invalid_navsat_geometry(navsat: pd.DataFrame) -> pd.DataFrame:
    """Drop NAV-SAT rows whose elevation/azimuth are out of range.

    u-blox NAV-SAT marks elevation as 'unknown' when outside [-90, 90] deg (e.g. the sentinel
    -91) and azimuth outside [0, 360] deg; such rows would otherwise yield a spurious
    ephemeris AoA (for example SV13 in this dataset).
    """
    elev = navsat["elev"].astype(float)
    azim = navsat["azim"].astype(float)
    return navsat.loc[elev.abs().le(90.0) & azim.between(0.0, 360.0)]


def compute_navsat_aoa(
    navsat: pd.DataFrame,
    baseline_azimuth_bias_deg: float | None = None,
) -> pd.DataFrame:
    """Compute receiver-reported NAV-SAT AoA diagnostic from azimuth/elevation."""
    if navsat.empty:
        return pd.DataFrame()

    bias = (
        0.0 if baseline_azimuth_bias_deg is None else float(baseline_azimuth_bias_deg)
    )
    data = navsat.dropna(subset=["rcvTow", "svid", "elev", "azim"]).copy()
    data = _drop_invalid_navsat_geometry(data)
    if data.empty:
        return pd.DataFrame()

    data["navsat_epoch_index"] = pd.factorize(data["rcvTow"], sort=True)[0]
    elevation_rad = np.deg2rad(data["elev"].astype(float))
    azimuth_rad = np.deg2rad(data["azim"].astype(float) - bias)
    cos_alpha = np.cos(elevation_rad) * np.cos(azimuth_rad)
    data["navsat_aoa_deg"] = np.rad2deg(np.arccos(np.clip(cos_alpha, -1.0, 1.0)))
    diagnostic = data.pivot_table(
        index="navsat_epoch_index",
        columns="svid",
        values="navsat_aoa_deg",
        aggfunc="first",
    )
    diagnostic.columns = [int(col) for col in diagnostic.columns]
    diagnostic.index.name = "epoch_index"
    return diagnostic.sort_index(axis=1)


def compute_reference_alpha_by_epoch(
    epochs: pd.DataFrame,
    navsat: pd.DataFrame,
    reference_svid: int,
    baseline_azimuth_bias_deg: float | None = None,
    navsat_tow_decimals: int = 0,
    epoch_tow_column: str = "rcvTow_rx1",
) -> pd.DataFrame:
    """Match a reference satellite's NAV-SAT AoA to synchronized RAWX epochs."""
    required_epoch_columns = {"epoch_index", epoch_tow_column}
    missing_epoch_columns = required_epoch_columns - set(epochs.columns)
    if missing_epoch_columns:
        raise ValueError(
            f"epochs is missing required columns: {sorted(missing_epoch_columns)}"
        )
    if navsat.empty:
        raise ValueError("NAV-SAT observations are required for reference AoA.")

    bias = (
        0.0 if baseline_azimuth_bias_deg is None else float(baseline_azimuth_bias_deg)
    )
    nav = navsat.loc[navsat["svid"].eq(int(reference_svid))].copy()
    nav = nav.dropna(subset=["rcvTow", "elev", "azim"])
    nav = _drop_invalid_navsat_geometry(nav)
    if nav.empty:
        raise ValueError(
            f"No valid NAV-SAT AoA rows found for reference SV {reference_svid}."
        )

    nav["_navsat_sync_tow"] = nav["rcvTow"].round(navsat_tow_decimals)
    nav = nav.sort_values("rcvTow").drop_duplicates("_navsat_sync_tow", keep="first")
    elevation_rad = np.deg2rad(nav["elev"].astype(float))
    azimuth_rad = np.deg2rad(nav["azim"].astype(float) - bias)
    cos_alpha = np.cos(elevation_rad) * np.cos(azimuth_rad)
    nav["reference_alpha_deg"] = np.rad2deg(
        np.arccos(np.clip(cos_alpha, -1.0, 1.0))
    )

    epoch_data = epochs[["epoch_index", epoch_tow_column]].copy()
    epoch_data["_navsat_sync_tow"] = epoch_data[epoch_tow_column].round(
        navsat_tow_decimals
    )
    matched = epoch_data.merge(
        nav[
            [
                "_navsat_sync_tow",
                "rcvTow",
                "svid",
                "elev",
                "azim",
                "cno",
                "svUsed",
                "reference_alpha_deg",
            ]
        ],
        how="left",
        on="_navsat_sync_tow",
    )
    matched = matched.rename(
        columns={
            "rcvTow": "navsat_rcvTow",
            "svid": "reference_svid",
            "elev": "reference_elev_deg",
            "azim": "reference_azim_deg",
            "cno": "reference_cno",
            "svUsed": "reference_svUsed",
        }
    )
    return matched.drop(columns=["_navsat_sync_tow"])


def longest_valid_epoch_segment(valid_mask: pd.Series) -> list[int]:
    """Return the longest contiguous epoch segment where a boolean mask is true."""
    segments: list[list[int]] = []
    current: list[int] = []
    previous: int | None = None

    for epoch, valid in valid_mask.items():
        epoch = int(epoch)
        if not bool(valid):
            if current:
                segments.append(current)
                current = []
            previous = None
            continue

        if previous is None or epoch == previous + 1:
            current.append(epoch)
        else:
            if current:
                segments.append(current)
            current = [epoch]
        previous = epoch

    if current:
        segments.append(current)
    return max(segments, key=len) if segments else []


def sort_svid_columns(
    frame: pd.DataFrame,
    selected_svids: Sequence[int] | None = None,
) -> pd.DataFrame:
    """Sort regular SVID columns numerically and optionally reindex to a subset."""
    result = frame.copy()
    result.columns = [int(column) for column in result.columns]
    result = result.sort_index(axis=1)
    selected = _normalise_svids(selected_svids)
    if selected is not None:
        result = result.reindex(columns=list(selected))
    result.index.name = "epoch_index"
    return result


def _navsat_epoch_merge(
    epochs: pd.DataFrame,
    navsat: pd.DataFrame,
    *,
    epoch_tow_column: str,
    navsat_tow_decimals: int,
) -> pd.DataFrame:
    required = {"epoch_index", epoch_tow_column}
    missing = required - set(epochs.columns)
    if missing:
        raise ValueError(f"epochs is missing required columns: {sorted(missing)}")

    epoch_data = epochs[["epoch_index", epoch_tow_column]].copy()
    epoch_data["_navsat_sync_tow"] = epoch_data[epoch_tow_column].round(
        navsat_tow_decimals
    )

    if navsat.empty:
        return epoch_data.assign(svid=np.nan)

    nav = navsat.dropna(subset=["rcvTow", "svid"]).copy()
    nav["svid"] = nav["svid"].astype(int)
    nav["_navsat_sync_tow"] = nav["rcvTow"].round(navsat_tow_decimals)
    nav = nav.sort_values("rcvTow").drop_duplicates(
        ["_navsat_sync_tow", "svid"], keep="first"
    )
    return epoch_data.merge(nav, how="left", on="_navsat_sync_tow")


def compute_navsat_aoa_by_synced_epoch(
    epochs: pd.DataFrame,
    navsat: pd.DataFrame,
    *,
    epoch_tow_column: str,
    navsat_tow_decimals: int = 0,
    baseline_azimuth_bias_deg: float | None = None,
    selected_svids: Sequence[int] | None = None,
) -> pd.DataFrame:
    """Compute NAV-SAT ephemeris-style AoA matched to synchronized RAWX epochs."""
    matched = _navsat_epoch_merge(
        epochs,
        navsat,
        epoch_tow_column=epoch_tow_column,
        navsat_tow_decimals=navsat_tow_decimals,
    )
    matched = matched.dropna(subset=["svid", "elev", "azim"]).copy()
    matched = _drop_invalid_navsat_geometry(matched)
    if matched.empty:
        return pd.DataFrame(index=epochs["epoch_index"].astype(int))

    bias = (
        0.0 if baseline_azimuth_bias_deg is None else float(baseline_azimuth_bias_deg)
    )
    elevation_rad = np.deg2rad(matched["elev"].astype(float))
    azimuth_rad = np.deg2rad(matched["azim"].astype(float) - bias)
    cos_alpha = np.cos(elevation_rad) * np.cos(azimuth_rad)
    matched["ephemeris_aoa_deg"] = np.rad2deg(
        np.arccos(np.clip(cos_alpha, -1.0, 1.0))
    )
    pivot = matched.pivot_table(
        index="epoch_index",
        columns="svid",
        values="ephemeris_aoa_deg",
        aggfunc="first",
    )
    pivot = pivot.reindex(epochs["epoch_index"].astype(int))
    return sort_svid_columns(pivot, selected_svids)


def compute_navsat_sv_used_by_synced_epoch(
    epochs: pd.DataFrame,
    navsat: pd.DataFrame,
    *,
    epoch_tow_column: str,
    navsat_tow_decimals: int = 0,
    selected_svids: Sequence[int] | None = None,
) -> pd.DataFrame:
    """Compute NAV-SAT svUsed matched to synchronized RAWX epochs."""
    matched = _navsat_epoch_merge(
        epochs,
        navsat,
        epoch_tow_column=epoch_tow_column,
        navsat_tow_decimals=navsat_tow_decimals,
    )
    matched = matched.dropna(subset=["svid", "svUsed"]).copy()
    if matched.empty:
        return pd.DataFrame(index=epochs["epoch_index"].astype(int))

    matched["svUsed"] = pd.to_numeric(matched["svUsed"], errors="coerce")
    pivot = matched.pivot_table(
        index="epoch_index",
        columns="svid",
        values="svUsed",
        aggfunc="max",
    )
    pivot = pivot.reindex(epochs["epoch_index"].astype(int))
    return sort_svid_columns(pivot, selected_svids)


def compute_navsat_sv_used(
    navsat: pd.DataFrame,
    selected_svids: Sequence[int] | None = None,
) -> pd.DataFrame:
    """Compute receiver-native NAV-SAT svUsed over every NAV-SAT epoch."""
    if navsat.empty:
        return pd.DataFrame()

    data = navsat.dropna(subset=["rcvTow", "svid", "svUsed"]).copy()
    if data.empty:
        return pd.DataFrame()

    data["svid"] = data["svid"].astype(int)
    data["svUsed"] = pd.to_numeric(data["svUsed"], errors="coerce")
    data["navsat_epoch_index"] = pd.factorize(data["rcvTow"], sort=True)[0]
    pivot = data.pivot_table(
        index="navsat_epoch_index",
        columns="svid",
        values="svUsed",
        aggfunc="max",
    )
    return sort_svid_columns(pivot, selected_svids)


def summarize_sv_used(
    sv_used_by_receiver: dict[str, pd.DataFrame],
    svids: Sequence[int],
) -> pd.DataFrame:
    """Summarize whether selected satellites were used in receiver navigation."""
    selected_svids = _normalise_svids(svids) or ()
    rows: list[dict[str, object]] = []
    for receiver, frame in sv_used_by_receiver.items():
        for svid in selected_svids:
            series = frame.get(int(svid), pd.Series(index=frame.index, dtype=float))
            observed = series.notna()
            used = series.fillna(0).astype(float).gt(0)
            used_epochs = series.index[used].to_list()
            rows.append(
                {
                    "receiver": receiver,
                    "svid": int(svid),
                    "total_epoch_count": int(len(series)),
                    "observed_epoch_count": int(observed.sum()),
                    "used_epoch_count": int((observed & used).sum()),
                    "pct_observed_epochs_used": float(
                        used.loc[observed].mean() * 100.0
                    )
                    if observed.any()
                    else float("nan"),
                    "first_used_epoch_index": int(used_epochs[0])
                    if used_epochs
                    else None,
                    "last_used_epoch_index": int(used_epochs[-1])
                    if used_epochs
                    else None,
                }
            )
    return pd.DataFrame(rows)


def select_joint_source_aoa(
    candidates: pd.DataFrame,
    svids: Sequence[int],
    spread_weight: float = 4.0,
    smooth_weight: float = 1.0,
    max_states_per_epoch: int = 50_000,
    return_branches: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Select AoA branches jointly for satellites expected to share one RF source."""
    required = {"epoch_index", "svid", "branch", "candidate_aoa_deg"}
    missing = required - set(candidates.columns)
    if missing:
        raise ValueError(f"candidates is missing required columns: {sorted(missing)}")

    selected_svids = _normalise_svids(svids) or ()
    if not selected_svids:
        raise ValueError("At least one SVID is required.")

    branch_values = sorted(int(branch) for branch in candidates["branch"].unique())
    if len(branch_values) ** len(selected_svids) > max_states_per_epoch:
        raise ValueError(
            "Too many branch combinations for joint AoA selection; reduce svids "
            "or increase max_states_per_epoch."
        )

    epochs = sorted(int(epoch) for epoch in candidates["epoch_index"].unique())
    states_by_epoch: list[list[dict[str, object]]] = []
    valid_epochs: list[int] = []
    for epoch in epochs:
        per_sat: dict[int, dict[int, float]] = {}
        for svid in selected_svids:
            subset = candidates.loc[
                candidates["epoch_index"].eq(epoch)
                & candidates["svid"].eq(svid)
                & candidates["candidate_aoa_deg"].notna()
            ]
            per_sat[svid] = {
                int(row.branch): float(row.candidate_aoa_deg)
                for row in subset.itertuples()
            }

        epoch_states: list[dict[str, object]] = []
        for branch_combo in product(branch_values, repeat=len(selected_svids)):
            values: list[float] = []
            for svid, branch in zip(selected_svids, branch_combo):
                if branch not in per_sat[svid]:
                    break
                values.append(per_sat[svid][branch])
            else:
                value_array = np.asarray(values, dtype=float)
                epoch_states.append(
                    {
                        "branches": tuple(int(branch) for branch in branch_combo),
                        "values": value_array,
                        "spread": float(value_array.max() - value_array.min()),
                    }
                )

        if epoch_states:
            states_by_epoch.append(epoch_states)
            valid_epochs.append(epoch)

    selected = pd.DataFrame(index=valid_epochs, columns=selected_svids, dtype=float)
    branches = pd.DataFrame(index=valid_epochs, columns=selected_svids, dtype=float)
    selected.index.name = "epoch_index"
    branches.index.name = "epoch_index"
    if not states_by_epoch:
        return (selected, branches) if return_branches else selected

    costs: list[np.ndarray] = [
        np.asarray(
            [spread_weight * float(state["spread"]) for state in states_by_epoch[0]],
            dtype=float,
        )
    ]
    backs: list[np.ndarray] = [np.full(len(states_by_epoch[0]), -1, dtype=int)]
    for epoch_index in range(1, len(states_by_epoch)):
        previous_states = states_by_epoch[epoch_index - 1]
        current_states = states_by_epoch[epoch_index]
        epoch_cost = np.full(len(current_states), np.inf, dtype=float)
        epoch_back = np.full(len(current_states), -1, dtype=int)
        for current_index, current_state in enumerate(current_states):
            current_values = current_state["values"]
            transition = np.asarray(
                [
                    np.mean(np.abs(current_values - previous_state["values"]))
                    for previous_state in previous_states
                ],
                dtype=float,
            )
            total = (
                costs[epoch_index - 1]
                + smooth_weight * transition
                + spread_weight * float(current_state["spread"])
            )
            best_previous = int(np.nanargmin(total))
            epoch_cost[current_index] = total[best_previous]
            epoch_back[current_index] = best_previous
        costs.append(epoch_cost)
        backs.append(epoch_back)

    path = [int(np.nanargmin(costs[-1]))]
    for epoch_index in range(len(states_by_epoch) - 1, 0, -1):
        path.append(int(backs[epoch_index][path[-1]]))
    path = list(reversed(path))

    for epoch, state_index, epoch_states in zip(valid_epochs, path, states_by_epoch):
        state = epoch_states[state_index]
        for svid, value, branch in zip(
            selected_svids, state["values"], state["branches"]
        ):
            selected.loc[epoch, svid] = float(value)
            branches.loc[epoch, svid] = int(branch)

    if return_branches:
        return selected, branches
    return selected


def compute_pairwise_aoa_convergence(
    selected_aoa: pd.DataFrame,
    threshold_deg: float = 5.0,
) -> pd.DataFrame:
    """Summarize pairwise AoA convergence for selected spoofing satellites."""
    columns = [int(column) for column in selected_aoa.columns]
    rows: list[dict[str, object]] = []
    for left_index, left_svid in enumerate(columns):
        for right_svid in columns[left_index + 1 :]:
            diff = (
                selected_aoa[left_svid] - selected_aoa[right_svid]
            ).abs().dropna()
            rows.append(
                {
                    "pair": f"{left_svid}-{right_svid}",
                    "left_svid": left_svid,
                    "right_svid": right_svid,
                    "epoch_count": int(len(diff)),
                    "median_abs_aoa_diff_deg": float(diff.median())
                    if not diff.empty
                    else float("nan"),
                    "mean_abs_aoa_diff_deg": float(diff.mean())
                    if not diff.empty
                    else float("nan"),
                    "p90_abs_aoa_diff_deg": float(diff.quantile(0.9))
                    if not diff.empty
                    else float("nan"),
                    "max_abs_aoa_diff_deg": float(diff.max())
                    if not diff.empty
                    else float("nan"),
                    "pct_epochs_aoa_diff_le_threshold": float(
                        (diff <= threshold_deg).mean() * 100.0
                    )
                    if not diff.empty
                    else float("nan"),
                    "threshold_deg": float(threshold_deg),
                }
            )
    return pd.DataFrame(rows)


def summarize_converged_aoa_clusters(
    selected_aoa: pd.DataFrame,
    threshold_deg: float = 5.0,
) -> pd.DataFrame:
    """Find connected AoA clusters and summarize their median angle."""
    columns = [int(column) for column in selected_aoa.columns]
    if not columns:
        return pd.DataFrame()

    pairwise = compute_pairwise_aoa_convergence(selected_aoa, threshold_deg)
    adjacency = {svid: {svid} for svid in columns}
    for row in pairwise.itertuples():
        if (
            pd.notna(row.median_abs_aoa_diff_deg)
            and float(row.median_abs_aoa_diff_deg) <= threshold_deg
        ):
            adjacency[int(row.left_svid)].add(int(row.right_svid))
            adjacency[int(row.right_svid)].add(int(row.left_svid))

    visited: set[int] = set()
    clusters: list[list[int]] = []
    for svid in columns:
        if svid in visited:
            continue
        stack = [svid]
        cluster: list[int] = []
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            cluster.append(current)
            stack.extend(sorted(adjacency[current] - visited))
        clusters.append(sorted(cluster))

    rows: list[dict[str, object]] = []
    for cluster in clusters:
        data = selected_aoa[cluster].dropna(how="all")
        epoch_median = data.median(axis=1, skipna=True)
        spread = data.max(axis=1, skipna=True) - data.min(axis=1, skipna=True)
        rows.append(
            {
                "cluster_svids": ",".join(str(svid) for svid in cluster),
                "cluster_size": int(len(cluster)),
                "epoch_count": int(epoch_median.notna().sum()),
                "median_aoa_deg": float(epoch_median.median(skipna=True)),
                "mean_aoa_deg": float(epoch_median.mean(skipna=True)),
                "std_epoch_median_aoa_deg": float(epoch_median.std(skipna=True)),
                "median_spread_deg": float(spread.median(skipna=True)),
                "p90_spread_deg": float(spread.quantile(0.9)),
                "threshold_deg": float(threshold_deg),
            }
        )
    return pd.DataFrame(rows).sort_values(
        by=["cluster_size", "epoch_count"], ascending=[False, False]
    )


def reconstruct_multi_reference_aoa(
    single_difference_cycles: pd.DataFrame,
    ephemeris_aoa_deg: pd.DataFrame,
    *,
    reference_svids: Sequence[int] | None = None,
    selected_svids: Sequence[int] | None = None,
    min_target_epochs: int = 10,
    branch_values: Iterable[int] = (-2, -1, 0, 1, 2),
    baseline_over_wavelength: float = 1.0,
    branch_selection_method: str = "dp",
) -> pd.DataFrame:
    """Reconstruct DD AoAs for every eligible DD reference.

    ``branch_selection_method="dp"`` applies temporal DP and backtracking without using target
    ephemeris to select an ambiguity branch. ``"best_m"`` instead selects, for every satellite,
    the constant branch with the smallest median AoA error to that satellite's ephemeris track.
    Best-m uses no authenticity labels, but is an ephemeris-assisted diagnostic rather than an
    independent spoof detector because the same ephemeris is later used in the residual.
    """
    if min_target_epochs < 1:
        raise ValueError("min_target_epochs must be at least 1.")
    if branch_selection_method not in {"dp", "best_m"}:
        raise ValueError("branch_selection_method must be 'dp' or 'best_m'.")

    single_difference = sort_svid_columns(single_difference_cycles)
    ephemeris = sort_svid_columns(ephemeris_aoa_deg).reindex(single_difference.index)
    available = sorted(set(single_difference.columns) & set(ephemeris.columns))
    requested_targets = _normalise_svids(selected_svids)
    if requested_targets is not None:
        available = [svid for svid in available if svid in requested_targets]

    eligible = [
        svid
        for svid in available
        if int(single_difference[svid].notna().sum()) >= min_target_epochs
        and int(ephemeris[svid].notna().sum()) >= min_target_epochs
    ]
    if len(eligible) < 3:
        raise ValueError(
            "At least three satellites with sufficient DD and ephemeris observations are "
            "required for pairwise geometry classification."
        )

    requested_references = _normalise_svids(reference_svids)
    if requested_references is None:
        references = eligible
    else:
        references = [svid for svid in requested_references if svid in eligible]
    if not references:
        raise ValueError("No requested DD reference has sufficient observations.")

    rows: list[pd.DataFrame] = []
    for reference_svid in references:
        double_difference = compute_double_difference(
            single_difference.reindex(columns=eligible), reference_svid
        )
        _, fractional_dd = fractional_round_dd(double_difference)
        reference_aoa = ephemeris[reference_svid]
        candidates = generate_aoa_candidates(
            fractional_dd,
            alpha_r_deg=reference_aoa,
            branch_values=branch_values,
            baseline_over_wavelength=baseline_over_wavelength,
        )
        if branch_selection_method == "dp":
            selected_aoa, selected_branch = select_aoa_dp(
                candidates,
                reference_svid=reference_svid,
                reference_angle_deg=reference_aoa,
                return_branches=True,
            )
        else:
            selected_aoa, selected_branch_by_svid = select_aoa_by_reference(
                candidates,
                ephemeris.reindex(columns=eligible),
                return_branches=True,
            )
            selected_branch = pd.DataFrame(
                np.nan,
                index=selected_aoa.index,
                columns=selected_aoa.columns,
                dtype=float,
            )
            selected_branch.index.name = "epoch_index"
            for svid, branch in selected_branch_by_svid.items():
                valid_epochs = selected_aoa[int(svid)].notna()
                selected_branch.loc[valid_epochs, int(svid)] = int(branch)

        aoa_long = selected_aoa.reset_index().melt(
            id_vars="epoch_index", var_name="svid", value_name="dd_aoa_deg"
        )
        branch_long = selected_branch.reset_index().melt(
            id_vars="epoch_index", var_name="svid", value_name="selected_branch"
        )
        ephemeris_long = ephemeris.reindex(columns=eligible).reset_index().melt(
            id_vars="epoch_index", var_name="svid", value_name="ephemeris_aoa_deg"
        )
        reference_rows = aoa_long.merge(
            branch_long, on=["epoch_index", "svid"], how="left"
        ).merge(ephemeris_long, on=["epoch_index", "svid"], how="left")
        reference_rows.insert(0, "reference_svid", int(reference_svid))
        reference_rows.insert(1, "branch_selection_method", branch_selection_method)
        reference_rows["svid"] = reference_rows["svid"].astype(int)
        reference_rows["dd_direction_cosine"] = np.cos(
            np.deg2rad(reference_rows["dd_aoa_deg"])
        )
        reference_rows["ephemeris_direction_cosine"] = np.cos(
            np.deg2rad(reference_rows["ephemeris_aoa_deg"])
        )
        reference_rows["signed_geometry_error"] = (
            reference_rows["dd_direction_cosine"]
            - reference_rows["ephemeris_direction_cosine"]
        )
        rows.append(reference_rows)

    return pd.concat(rows, ignore_index=True).sort_values(
        ["reference_svid", "epoch_index", "svid"]
    ).reset_index(drop=True)


def wrap_cycles(values: float | np.ndarray | pd.Series) -> float | np.ndarray | pd.Series:
    """Wrap carrier-cycle values to the half-open interval [-0.5, 0.5)."""
    return (values + 0.5) % 1.0 - 0.5


def compute_ephemeris_predicted_pair_separation_cycles(
    left_aoa_deg: float | np.ndarray | pd.Series,
    right_aoa_deg: float | np.ndarray | pd.Series,
    *,
    baseline_over_wavelength: float,
) -> float | np.ndarray | pd.Series:
    """Project two ephemeris AoAs into observable fractional carrier separation."""
    if baseline_over_wavelength <= 0:
        raise ValueError("baseline_over_wavelength must be positive.")
    left_direction_cosine = np.cos(np.deg2rad(left_aoa_deg))
    right_direction_cosine = np.cos(np.deg2rad(right_aoa_deg))
    predicted_pair_cycles = baseline_over_wavelength * (
        left_direction_cosine - right_direction_cosine
    )
    return abs(wrap_cycles(predicted_pair_cycles))


def compute_wrapped_pairwise_residuals(
    single_difference_cycles: pd.DataFrame,
    ephemeris_aoa_deg: pd.DataFrame,
    *,
    selected_svids: Sequence[int] | None = None,
    min_target_epochs: int = 10,
    baseline_over_wavelength: float = 1.0,
) -> pd.DataFrame:
    """Compute branch-invariant pairwise geometry and common-source residuals.

    Subtracting two receiver single differences gives an inter-satellite double difference.
    Wrapping that measurement-minus-ephemeris difference modulo one cycle removes the unknown
    integer ambiguity exactly, so neither temporal DP nor ephemeris-aligned best-m is needed.
    """
    if min_target_epochs < 1:
        raise ValueError("min_target_epochs must be at least 1.")
    if baseline_over_wavelength <= 0:
        raise ValueError("baseline_over_wavelength must be positive.")

    single_difference = sort_svid_columns(single_difference_cycles)
    ephemeris = sort_svid_columns(ephemeris_aoa_deg).reindex(single_difference.index)
    available = sorted(set(single_difference.columns) & set(ephemeris.columns))
    requested_svids = _normalise_svids(selected_svids)
    if requested_svids is not None:
        available = [svid for svid in available if svid in requested_svids]
    eligible = [
        int(svid)
        for svid in available
        if int(single_difference[svid].notna().sum()) >= min_target_epochs
        and int(ephemeris[svid].notna().sum()) >= min_target_epochs
    ]
    if len(eligible) < 2:
        raise ValueError(
            "At least two satellites with sufficient single-difference and ephemeris "
            "observations are required."
        )

    ephemeris_direction_cosine = np.cos(np.deg2rad(ephemeris.reindex(columns=eligible)))
    rows: list[pd.DataFrame] = []
    for left_svid, right_svid in combinations(eligible, 2):
        pair = pd.DataFrame(
            {
                "measured_pair_cycles": (
                    single_difference[left_svid] - single_difference[right_svid]
                ),
                "expected_pair_cycles": baseline_over_wavelength
                * (
                    ephemeris_direction_cosine[left_svid]
                    - ephemeris_direction_cosine[right_svid]
                ),
            }
        ).dropna()
        if pair.empty:
            continue
        pair["measured_pair_cycles_wrapped"] = wrap_cycles(
            pair["measured_pair_cycles"]
        )
        pair["expected_pair_cycles_wrapped"] = wrap_cycles(
            pair["expected_pair_cycles"]
        )
        pair["signed_wrapped_geometry_residual_cycles"] = wrap_cycles(
            pair["measured_pair_cycles"] - pair["expected_pair_cycles"]
        )
        pair["wrapped_geometry_residual_cycles"] = pair[
            "signed_wrapped_geometry_residual_cycles"
        ].abs()
        pair["common_source_collapse_cycles"] = pair[
            "measured_pair_cycles_wrapped"
        ].abs()
        pair["wrapped_expected_separation_cycles"] = pair[
            "expected_pair_cycles_wrapped"
        ].abs()
        pair["common_source_incompatibility_cycles"] = (
            pair["wrapped_expected_separation_cycles"]
            - pair["common_source_collapse_cycles"]
        )
        pair = pair.reset_index().rename(columns={pair.index.name or "index": "epoch_index"})
        pair.insert(1, "left_svid", int(left_svid))
        pair.insert(2, "right_svid", int(right_svid))
        rows.append(pair)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).sort_values(
        ["left_svid", "right_svid", "epoch_index"]
    ).reset_index(drop=True)


def aggregate_wrapped_pairwise_residuals(
    epoch_residuals: pd.DataFrame,
    *,
    min_epochs_per_pair: int = 10,
) -> pd.DataFrame:
    """Aggregate branch-invariant pairwise residuals over epochs."""
    if min_epochs_per_pair < 1:
        raise ValueError("min_epochs_per_pair must be at least 1.")
    required = {
        "left_svid",
        "right_svid",
        "wrapped_geometry_residual_cycles",
        "common_source_collapse_cycles",
        "wrapped_expected_separation_cycles",
        "common_source_incompatibility_cycles",
    }
    missing = required - set(epoch_residuals.columns)
    if missing:
        raise ValueError(f"epoch_residuals is missing required columns: {sorted(missing)}")
    if epoch_residuals.empty:
        return pd.DataFrame()

    aggregated = (
        epoch_residuals.groupby(["left_svid", "right_svid"], as_index=False)
        .agg(
            median_wrapped_geometry_residual=(
                "wrapped_geometry_residual_cycles",
                "median",
            ),
            mean_wrapped_geometry_residual=(
                "wrapped_geometry_residual_cycles",
                "mean",
            ),
            p90_wrapped_geometry_residual=(
                "wrapped_geometry_residual_cycles",
                lambda values: values.quantile(0.9),
            ),
            median_common_source_collapse=(
                "common_source_collapse_cycles",
                "median",
            ),
            mean_common_source_collapse=(
                "common_source_collapse_cycles",
                "mean",
            ),
            p90_common_source_collapse=(
                "common_source_collapse_cycles",
                lambda values: values.quantile(0.9),
            ),
            median_expected_separation=(
                "wrapped_expected_separation_cycles",
                "median",
            ),
            median_common_source_incompatibility=(
                "common_source_incompatibility_cycles",
                "median",
            ),
            epoch_count=("wrapped_geometry_residual_cycles", "count"),
        )
        .loc[lambda frame: frame["epoch_count"].ge(min_epochs_per_pair)]
        .reset_index(drop=True)
    )
    aggregated["pair"] = aggregated.apply(
        lambda row: f"PRN {int(row.left_svid)}–PRN {int(row.right_svid)}", axis=1
    )
    return aggregated


def find_pairwise_cliques(
    aggregated_pairs: pd.DataFrame,
    *,
    edge_value_column: str,
    edge_threshold: float,
    min_clique_size: int = 3,
) -> pd.DataFrame:
    """Find maximal complete subgraphs whose pair values do not exceed a threshold."""
    if edge_threshold < 0:
        raise ValueError("edge_threshold must be non-negative.")
    if min_clique_size < 2:
        raise ValueError("min_clique_size must be at least 2.")
    required = {"left_svid", "right_svid", edge_value_column}
    missing = required - set(aggregated_pairs.columns)
    if missing:
        raise ValueError(f"aggregated_pairs is missing required columns: {sorted(missing)}")

    edge_values = {
        tuple(sorted((int(row.left_svid), int(row.right_svid)))): float(
            getattr(row, edge_value_column)
        )
        for row in aggregated_pairs.itertuples()
        if np.isfinite(float(getattr(row, edge_value_column)))
    }
    svids = sorted({svid for edge in edge_values for svid in edge})
    complete_subsets: list[tuple[int, ...]] = []
    for size in range(min_clique_size, len(svids) + 1):
        for subset in combinations(svids, size):
            if all(
                edge_values.get((left_svid, right_svid), np.inf) <= edge_threshold
                for left_svid, right_svid in combinations(subset, 2)
            ):
                complete_subsets.append(tuple(int(svid) for svid in subset))

    maximal_subsets = [
        subset
        for subset in complete_subsets
        if not any(set(subset) < set(other) for other in complete_subsets)
    ]
    rows: list[dict[str, object]] = []
    for subset in maximal_subsets:
        values = [
            edge_values[(left_svid, right_svid)]
            for left_svid, right_svid in combinations(subset, 2)
        ]
        rows.append(
            {
                "clique_svids": subset,
                "clique_size": len(subset),
                "edge_count": len(values),
                "median_edge_value": float(np.median(values)),
                "max_edge_value": float(np.max(values)),
                "edge_threshold": float(edge_threshold),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "clique_svids",
                "clique_size",
                "edge_count",
                "median_edge_value",
                "max_edge_value",
                "edge_threshold",
            ]
        )
    return pd.DataFrame(rows).sort_values(
        ["clique_size", "median_edge_value"], ascending=[False, True]
    ).reset_index(drop=True)


def compute_pairwise_geometry_residuals(
    geometry_tracks: pd.DataFrame,
    *,
    residual_domain: str = "direction_cosine",
) -> pd.DataFrame:
    """Compute pairwise geometry residuals per epoch and DD reference.

    ``residual_domain="direction_cosine"`` reproduces journal Eq. 32. ``"degrees"``
    applies the same difference-of-relative-geometry construction directly to the reconstructed
    and ephemeris AoAs. These AoAs are arccos outputs in [0, 180] degrees, so circular wrapping
    is not required.
    """
    if residual_domain == "direction_cosine":
        measured_column = "dd_direction_cosine"
        expected_column = "ephemeris_direction_cosine"
        measured_output = "measured_relative_direction_cosine"
        expected_output = "expected_relative_direction_cosine"
        residual_unit = "direction_cosine"
    elif residual_domain == "degrees":
        measured_column = "dd_aoa_deg"
        expected_column = "ephemeris_aoa_deg"
        measured_output = "measured_relative_aoa_deg"
        expected_output = "expected_relative_aoa_deg"
        residual_unit = "deg"
    else:
        raise ValueError(
            "residual_domain must be 'direction_cosine' or 'degrees'."
        )

    required = {
        "reference_svid",
        "epoch_index",
        "svid",
        measured_column,
        expected_column,
    }
    missing = required - set(geometry_tracks.columns)
    if missing:
        raise ValueError(f"geometry_tracks is missing required columns: {sorted(missing)}")

    all_svids = sorted(int(svid) for svid in geometry_tracks["svid"].unique())
    rows: list[pd.DataFrame] = []
    for reference_svid, reference_rows in geometry_tracks.groupby(
        "reference_svid", sort=True
    ):
        reference_svid = int(reference_svid)
        targets = [svid for svid in all_svids if svid != reference_svid]
        for left_svid, right_svid in combinations(targets, 2):
            left = reference_rows.loc[
                reference_rows["svid"].eq(left_svid),
                ["epoch_index", measured_column, expected_column],
            ].set_index("epoch_index")
            right = reference_rows.loc[
                reference_rows["svid"].eq(right_svid),
                ["epoch_index", measured_column, expected_column],
            ].set_index("epoch_index")
            paired = left.join(right, how="inner", lsuffix="_left", rsuffix="_right")
            paired = paired.dropna()
            if paired.empty:
                continue
            measured_relative = (
                paired[f"{measured_column}_left"]
                - paired[f"{measured_column}_right"]
            )
            expected_relative = (
                paired[f"{expected_column}_left"]
                - paired[f"{expected_column}_right"]
            )
            pair_rows = pd.DataFrame(
                {
                    "reference_svid": reference_svid,
                    "epoch_index": paired.index.astype(int),
                    "left_svid": left_svid,
                    "right_svid": right_svid,
                    "residual_domain": residual_domain,
                    "residual_unit": residual_unit,
                    measured_output: measured_relative,
                    expected_output: expected_relative,
                    "pairwise_residual": (measured_relative - expected_relative).abs(),
                }
            )
            rows.append(pair_rows.reset_index(drop=True))

    columns = [
        "reference_svid",
        "epoch_index",
        "left_svid",
        "right_svid",
        "residual_domain",
        "residual_unit",
        measured_output,
        expected_output,
        "pairwise_residual",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.concat(rows, ignore_index=True).sort_values(
        ["left_svid", "right_svid", "reference_svid", "epoch_index"]
    ).reset_index(drop=True)


def aggregate_pairwise_geometry_residuals(
    epoch_residuals: pd.DataFrame,
    *,
    min_epochs_per_reference: int = 10,
    min_references_per_pair: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply journal Eqs. 33-34: median over epochs, then valid DD references."""
    if min_epochs_per_reference < 1 or min_references_per_pair < 1:
        raise ValueError("Minimum epoch and reference counts must be at least 1.")
    required = {
        "reference_svid",
        "left_svid",
        "right_svid",
        "pairwise_residual",
    }
    missing = required - set(epoch_residuals.columns)
    if missing:
        raise ValueError(f"epoch_residuals is missing required columns: {sorted(missing)}")
    if epoch_residuals.empty:
        return pd.DataFrame(), pd.DataFrame()

    residual_metadata: dict[str, object] = {}
    for column in ("residual_domain", "residual_unit"):
        if column not in epoch_residuals:
            continue
        values = epoch_residuals[column].dropna().unique()
        if len(values) != 1:
            raise ValueError(f"epoch_residuals must contain exactly one {column} value.")
        residual_metadata[column] = values[0]

    by_reference = (
        epoch_residuals.groupby(
            ["reference_svid", "left_svid", "right_svid"], as_index=False
        )["pairwise_residual"]
        .agg(
            temporal_median_residual="median",
            mean_residual="mean",
            p90_residual=lambda values: values.quantile(0.9),
            epoch_count="count",
        )
        .loc[lambda frame: frame["epoch_count"].ge(min_epochs_per_reference)]
        .reset_index(drop=True)
    )
    if by_reference.empty:
        return by_reference, pd.DataFrame()
    for position, (column, value) in enumerate(residual_metadata.items(), start=3):
        by_reference.insert(position, column, value)

    aggregated = (
        by_reference.groupby(["left_svid", "right_svid"], as_index=False)
        .agg(
            aggregated_pairwise_residual=("temporal_median_residual", "median"),
            min_reference_residual=("temporal_median_residual", "min"),
            max_reference_residual=("temporal_median_residual", "max"),
            reference_count=("reference_svid", "count"),
            epoch_observation_count=("epoch_count", "sum"),
        )
        .loc[lambda frame: frame["reference_count"].ge(min_references_per_pair)]
        .reset_index(drop=True)
    )
    aggregated["pair"] = aggregated.apply(
        lambda row: f"PRN {int(row.left_svid)}–PRN {int(row.right_svid)}", axis=1
    )
    for position, (column, value) in enumerate(residual_metadata.items(), start=2):
        aggregated.insert(position, column, value)
    return by_reference, aggregated


def calibrate_pairwise_threshold(
    authentic_aggregated_residuals: pd.DataFrame,
    *,
    percentile: float = 95.0,
) -> float:
    """Calibrate journal threshold tau_p from an independent authentic dataset."""
    if not 0.0 < percentile < 100.0:
        raise ValueError("percentile must be between 0 and 100.")
    column = "aggregated_pairwise_residual"
    if column not in authentic_aggregated_residuals:
        raise ValueError(f"authentic_aggregated_residuals must contain {column!r}.")
    values = authentic_aggregated_residuals[column].dropna().astype(float)
    if values.empty:
        raise ValueError("No authentic pairwise residuals are available for calibration.")
    return float(values.quantile(percentile / 100.0))


def estimate_pairwise_threshold_gmm(
    aggregated_residuals: pd.DataFrame,
    *,
    value_column: str = "aggregated_pairwise_residual",
    random_state: int = 0,
) -> tuple[float, pd.DataFrame]:
    """Estimate an exploratory threshold from a two-component residual mixture.

    This is a label-free diagnostic for datasets without an authentic calibration recording;
    it is not equivalent to the journal's calibrated ``tau_p``. The returned threshold is the
    intersection of the two weighted Gaussian densities between their fitted means.
    """
    from sklearn.mixture import GaussianMixture

    if value_column not in aggregated_residuals:
        raise ValueError(f"aggregated_residuals must contain {value_column!r}.")
    values = (
        aggregated_residuals[value_column]
        .dropna()
        .astype(float)
        .to_numpy()
        .reshape(-1, 1)
    )
    if len(values) < 6:
        raise ValueError("At least six pairwise residuals are required for GMM estimation.")

    one_component = GaussianMixture(
        n_components=1,
        covariance_type="full",
        n_init=20,
        random_state=random_state,
        reg_covar=1e-6,
    ).fit(values)
    two_component = GaussianMixture(
        n_components=2,
        covariance_type="full",
        n_init=20,
        random_state=random_state,
        reg_covar=1e-6,
    ).fit(values)

    order = np.argsort(two_component.means_.ravel())
    means = two_component.means_.ravel()[order]
    standard_deviations = np.sqrt(two_component.covariances_.reshape(-1))[order]
    weights = two_component.weights_.ravel()[order]
    grid = np.linspace(float(means[0]), float(means[1]), 20_001)
    densities = []
    for mean, standard_deviation, weight in zip(means, standard_deviations, weights):
        density = (
            weight
            / (standard_deviation * np.sqrt(2.0 * np.pi))
            * np.exp(-0.5 * ((grid - mean) / standard_deviation) ** 2)
        )
        densities.append(density)
    threshold = float(grid[int(np.argmin(np.abs(densities[0] - densities[1])))])

    bic_one = float(one_component.bic(values))
    bic_two = float(two_component.bic(values))
    diagnostics = pd.DataFrame(
        [
            {
                "threshold_mode": "unsupervised_gmm_exploratory",
                "value_column": value_column,
                "pair_count": int(len(values)),
                "estimated_threshold": threshold,
                "low_component_mean": float(means[0]),
                "low_component_std": float(standard_deviations[0]),
                "low_component_weight": float(weights[0]),
                "high_component_mean": float(means[1]),
                "high_component_std": float(standard_deviations[1]),
                "high_component_weight": float(weights[1]),
                "one_component_bic": bic_one,
                "two_component_bic": bic_two,
                "bic_improvement_for_two_components": bic_one - bic_two,
            }
        ]
    )
    return threshold, diagnostics


def _symmetric_pairwise_matrix(
    aggregated_residuals: pd.DataFrame,
    value_column: str,
    *,
    diagonal: float | None = None,
) -> pd.DataFrame:
    svids = sorted(
        set(aggregated_residuals["left_svid"].astype(int))
        | set(aggregated_residuals["right_svid"].astype(int))
    )
    matrix = pd.DataFrame(np.nan, index=svids, columns=svids, dtype=float)
    matrix.index.name = "svid"
    matrix.columns.name = "svid"
    for row in aggregated_residuals.itertuples():
        left_svid = int(row.left_svid)
        right_svid = int(row.right_svid)
        value = float(getattr(row, value_column))
        matrix.loc[left_svid, right_svid] = value
        matrix.loc[right_svid, left_svid] = value
    if diagonal is not None:
        for svid in svids:
            matrix.loc[svid, svid] = float(diagonal)
    return matrix


def classify_pairwise_geometry(
    aggregated_residuals: pd.DataFrame,
    *,
    pairwise_threshold: float,
    score_threshold: float = 0.5,
    min_pair_coverage: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply journal Eqs. 37-39 and return scores, residual matrix, and A_ij."""
    if pairwise_threshold < 0:
        raise ValueError("pairwise_threshold must be non-negative.")
    if not 0.0 <= score_threshold <= 1.0:
        raise ValueError("score_threshold must be between 0 and 1.")
    if not 0.0 <= min_pair_coverage <= 1.0:
        raise ValueError("min_pair_coverage must be between 0 and 1.")
    required = {"left_svid", "right_svid", "aggregated_pairwise_residual"}
    missing = required - set(aggregated_residuals.columns)
    if missing:
        raise ValueError(
            f"aggregated_residuals is missing required columns: {sorted(missing)}"
        )
    if aggregated_residuals.empty:
        raise ValueError("No aggregated residuals are available for classification.")

    residual_matrix = _symmetric_pairwise_matrix(
        aggregated_residuals, "aggregated_pairwise_residual", diagonal=0.0
    )
    consistency_matrix = residual_matrix.le(float(pairwise_threshold)).astype(float)
    consistency_matrix = consistency_matrix.where(residual_matrix.notna())
    for svid in consistency_matrix.index:
        consistency_matrix.loc[svid, svid] = np.nan

    possible_pairs = max(len(residual_matrix) - 1, 1)
    rows: list[dict[str, object]] = []
    for svid in residual_matrix.index:
        valid = consistency_matrix.loc[svid].dropna()
        valid_pair_count = int(len(valid))
        consistent_pair_count = int(valid.eq(1.0).sum())
        coverage = valid_pair_count / possible_pairs
        consistency_score = consistent_pair_count / possible_pairs
        available_pair_score = (
            consistent_pair_count / valid_pair_count
            if valid_pair_count
            else float("nan")
        )
        if coverage < min_pair_coverage:
            decision = "Undetermined"
        elif consistency_score >= score_threshold:
            decision = "Authentic"
        else:
            decision = "Counterfeit"
        rows.append(
            {
                "svid": int(svid),
                "consistent_pair_count": consistent_pair_count,
                "valid_pair_count": valid_pair_count,
                "possible_pair_count": possible_pairs,
                "pair_coverage": coverage,
                "consistency_score": consistency_score,
                "available_pair_score": available_pair_score,
                "pairwise_threshold": float(pairwise_threshold),
                "score_threshold": float(score_threshold),
                "decision": decision,
            }
        )
    scores = pd.DataFrame(rows).sort_values(
        ["consistency_score", "svid"], ascending=[False, True]
    ).reset_index(drop=True)
    return scores, residual_matrix, consistency_matrix


def run_dataset(config: DatasetConfig) -> AoaRunResult:
    """Run one DD-method AoA analysis from UBX files through collapse metrics."""
    read_limit = None
    if config.length is not None:
        read_limit = config.start_index + config.length + 120
    obs1 = read_ubx_observations(
        config.receiver1_path,
        config.gnss_id,
        config.sig_id,
        max_rawx_epochs=read_limit,
    )
    obs2 = read_ubx_observations(
        config.receiver2_path,
        config.gnss_id,
        config.sig_id,
        max_rawx_epochs=read_limit,
    )
    synced = sync_receiver_epochs(
        obs1, obs2, tow_round_decimals=config.sync_tow_decimals
    )
    synced = _slice_synced_epochs(synced, config.start_index, config.length)

    carrier_phase_full = build_carrier_phase_table(
        synced,
        use_navpvt_time_correction=config.use_navpvt_time_correction,
        navpvt_tow_decimals=config.navpvt_tow_decimals,
    )
    reference_svid, reference_selection = select_reference_svid(
        carrier_phase_full,
        cno=carrier_phase_full.mean_cno,
        override=config.reference_svid,
        exclude_svids=config.excluded_reference_svids,
        required_svids=config.reference_required_svids,
    )

    selected_svids = _normalise_svids(config.selected_svids)
    excluded_analysis_svids = set(_normalise_svids(config.excluded_analysis_svids) or ())
    if selected_svids is None:
        analysis_svids = tuple(
            int(svid)
            for svid in carrier_phase_full.rx1_cycles.columns
            if int(svid) not in excluded_analysis_svids
        )
    else:
        analysis_svids = tuple(
            svid for svid in selected_svids if svid not in excluded_analysis_svids
        )
    phase_svids = tuple(dict.fromkeys((reference_svid, *analysis_svids)))
    carrier_phase = _filter_carrier_phase_columns(carrier_phase_full, phase_svids)

    single_difference = compute_single_difference(
        carrier_phase, order=config.single_difference_order
    )
    double_difference_full = compute_double_difference(
        single_difference, reference_svid
    )
    double_difference = double_difference_full.reindex(columns=list(analysis_svids))
    rounded_dd, fractional_dd = fractional_round_dd(double_difference)
    metrics_by_epoch, metrics_summary = compute_relative_dd_collapse_metrics(
        fractional_dd
    )

    if config.alpha_r_deg is None:
        candidates = pd.DataFrame(
            columns=[
                "epoch_index",
                "svid",
                "fractional_dd_cycles",
                "branch",
                "reference_alpha_deg",
                "candidate_cos_alpha",
                "candidate_aoa_deg",
                "valid_candidate",
            ]
        )
        selected_aoa = pd.DataFrame(index=fractional_dd.index)
        selected_branch = pd.DataFrame(index=fractional_dd.index)
        aoa_metrics_by_epoch = pd.DataFrame(index=fractional_dd.index)
        aoa_metrics_summary = pd.DataFrame()
    else:
        candidates = generate_aoa_candidates(
            fractional_dd,
            alpha_r_deg=config.alpha_r_deg,
            branch_values=config.branch_values,
            baseline_over_wavelength=config.baseline_over_wavelength,
        )
        selected_aoa, selected_branch = select_aoa_dp(
            candidates,
            reference_svid=reference_svid,
            reference_angle_deg=config.alpha_r_deg,
            return_branches=True,
        )
        aoa_metrics_by_epoch, aoa_metrics_summary = compute_collapse_metrics(
            selected_aoa
        )

    navsat_aoa = compute_navsat_aoa(
        obs1.navsat,
        baseline_azimuth_bias_deg=config.navsat_azimuth_bias_deg,
    )
    ephemeris_aoa = compute_navsat_aoa_by_synced_epoch(
        synced.epochs,
        obs1.navsat,
        epoch_tow_column="rcvTow_rx1",
        navsat_tow_decimals=config.navpvt_tow_decimals,
        baseline_azimuth_bias_deg=config.navsat_azimuth_bias_deg,
    )

    return AoaRunResult(
        config=config,
        epochs=synced.epochs,
        reference_svid=reference_svid,
        reference_selection=reference_selection,
        carrier_phase_cycles=carrier_phase.combined_cycles,
        carrier_phase_rx1_cycles=carrier_phase.rx1_cycles,
        carrier_phase_rx2_cycles=carrier_phase.rx2_cycles,
        single_difference_cycles=single_difference,
        double_difference_cycles=double_difference,
        rounded_dd_cycles=rounded_dd,
        fractional_dd_cycles=fractional_dd,
        candidate_aoa_deg=candidates,
        selected_aoa_deg=selected_aoa,
        selected_branch=selected_branch,
        collapse_metrics_by_epoch=metrics_by_epoch,
        collapse_metrics=metrics_summary,
        aoa_collapse_metrics_by_epoch=aoa_metrics_by_epoch,
        aoa_collapse_metrics=aoa_metrics_summary,
        navsat_aoa_diagnostic_deg=navsat_aoa,
        ephemeris_aoa_deg=ephemeris_aoa,
    )


def _write_csv(df: pd.DataFrame, path: Path, index: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index, na_rep="")


def _import_pyplot():
    import matplotlib.pyplot as plt

    return plt


def _plot_wide_dataframe(
    df: pd.DataFrame,
    title: str | None,
    ylabel: str,
    *,
    ylim: tuple[float, float] | None = None,
    highlight_svids: Sequence[int] | None = None,
    highlight_label: str | None = None,
):
    plt = _import_pyplot()
    fig, ax = plt.subplots(figsize=(12, 5))
    highlighted = set(_normalise_svids(highlight_svids) or ())
    for column in df.columns:
        series = df[column]
        if series.notna().any():
            svid = int(column)
            is_highlighted = svid in highlighted
            suffix = (
                f" ({highlight_label})"
                if is_highlighted and highlight_label
                else ""
            )
            ax.plot(
                df.index,
                series,
                label=f"PRN {svid}{suffix}",
                linewidth=2.0 if is_highlighted else 1.25,
                alpha=0.95 if is_highlighted else 0.8,
                marker="o" if is_highlighted else None,
                markersize=2.5,
                markevery=3,
            )
    if title:
        ax.set_title(title)
    ax.set_xlabel("Epoch index")
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.3)
    if len(df.columns) > 0:
        ax.legend(ncol=4, fontsize=8)
    fig.tight_layout()
    return fig


def plot_carrier_phase(result: AoaRunResult):
    plt = _import_pyplot()
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for ax, df, label in [
        (axes[0], result.carrier_phase_rx1_cycles, "Receiver 1"),
        (axes[1], result.carrier_phase_rx2_cycles, "Receiver 2"),
    ]:
        for column in df.columns:
            series = df[column]
            if series.notna().any():
                ax.plot(
                    df.index, series, label=f"SV {column}", linewidth=1.0, alpha=0.8
                )
        ax.set_ylabel("Carrier phase (cycles)")
        ax.set_title(label)
        ax.grid(True, alpha=0.3)
    axes[1].set_xlabel("Epoch index")
    axes[0].legend(ncol=4, fontsize=8)
    fig.tight_layout()
    return fig


def plot_single_difference(result: AoaRunResult):
    return _plot_wide_dataframe(
        result.single_difference_cycles,
        "Receiver Single Difference Carrier Phase",
        "Single difference (cycles)",
    )


def plot_double_difference(result: AoaRunResult):
    return _plot_wide_dataframe(
        result.double_difference_cycles,
        f"Double Difference Carrier Phase (reference SV {result.reference_svid})",
        "Double difference (cycles)",
    )


def plot_rounding_steps(result: AoaRunResult):
    return _plot_wide_dataframe(
        result.rounded_dd_cycles,
        "Rounded Double Difference Carrier Phase",
        "Rounded DD (cycles)",
    )


def plot_fractional_dd(
    result: AoaRunResult,
    *,
    highlight_svids: Sequence[int] | None = None,
    title: str | None = "Fractional Double Difference Carrier Phase",
):
    return _plot_wide_dataframe(
        result.fractional_dd_cycles,
        title,
        "Fractional DD (cycles)",
        ylim=(-0.55, 0.55),
        highlight_svids=highlight_svids,
        highlight_label="spoof",
    )


def plot_candidate_cloud(result: AoaRunResult):
    plt = _import_pyplot()
    fig, ax = plt.subplots(figsize=(12, 5))
    data = result.candidate_aoa_deg.dropna(subset=["candidate_aoa_deg"])
    if not data.empty:
        scatter = ax.scatter(
            data["epoch_index"],
            data["candidate_aoa_deg"],
            c=data["svid"],
            s=8,
            alpha=0.35,
            cmap="tab20",
        )
        fig.colorbar(scatter, ax=ax, label="SVID")
    ax.set_title("AoA Branch Candidates")
    ax.set_xlabel("Epoch index")
    ax.set_ylabel("Candidate AoA (deg)")
    ax.set_ylim(0, 180)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_valid_m_candidates(
    candidates: pd.DataFrame,
    *,
    reference_aoa: pd.DataFrame | None = None,
    title: str = "Valid AoA Candidates by Branch",
    highlight_svids: Sequence[int] = (),
):
    """Plot all physically valid AoA branches, faceted by SVID."""
    plt = _import_pyplot()
    from matplotlib.lines import Line2D

    data = valid_m_candidates(candidates)
    svids = sorted(int(svid) for svid in data["svid"].unique()) if not data.empty else []
    if not svids:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.set_title(title)
        ax.text(0.5, 0.5, "No valid candidates", ha="center", va="center")
        ax.axis("off")
        return fig

    ncols = min(3, len(svids))
    nrows = int(np.ceil(len(svids) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.8 * ncols, 3.2 * nrows),
        sharex=True,
        sharey=True,
    )
    axes_array = np.atleast_1d(axes).ravel()
    branch_colors = {
        -2: "#7f3b08",
        -1: "#2c7fb8",
        0: "#31a354",
        1: "#f16913",
        2: "#756bb1",
    }
    highlight_set = set(int(svid) for svid in highlight_svids)

    for ax, svid in zip(axes_array, svids):
        subset = data.loc[data["svid"].eq(svid)]
        for branch, branch_subset in subset.groupby("branch", sort=True):
            ax.scatter(
                branch_subset["epoch_index"],
                branch_subset["candidate_aoa_deg"],
                s=10,
                alpha=0.55,
                color=branch_colors.get(int(branch), "#636363"),
                label=f"m={int(branch)}",
            )
        if reference_aoa is not None and svid in reference_aoa.columns:
            reference = reference_aoa[svid].dropna()
            if not reference.empty:
                ax.plot(
                    reference.index,
                    reference,
                    color="black",
                    linestyle="--",
                    linewidth=1.4,
                    label="ephemeris",
                )
        suffix = " spoof" if svid in highlight_set else ""
        ax.set_title(f"SV {svid}{suffix}")
        ax.set_ylim(0, 180)
        ax.grid(True, alpha=0.3)

    for ax in axes_array[len(svids) :]:
        ax.axis("off")
    branch_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            label=f"m={int(branch)}",
            markerfacecolor=branch_colors.get(int(branch), "#636363"),
            markersize=6,
            alpha=0.7,
        )
        for branch in sorted(int(branch) for branch in data["branch"].unique())
    ]
    if reference_aoa is not None:
        branch_handles.append(
            Line2D(
                [0],
                [0],
                color="black",
                linestyle="--",
                linewidth=1.4,
                label="ephemeris",
            )
        )
    fig.legend(branch_handles, [handle.get_label() for handle in branch_handles], ncol=6)
    fig.suptitle(title, y=0.995)
    fig.supxlabel("Epoch index")
    fig.supylabel("AoA (deg)")
    fig.tight_layout()
    return fig


def plot_aoa_method_comparison(
    dd_aoa: pd.DataFrame,
    ephemeris_aoa: pd.DataFrame,
    *,
    title: str,
    highlight_svids: Sequence[int] = (),
):
    """Plot DD-method AoA against ephemeris AoA for each shared SVID."""
    plt = _import_pyplot()
    shared_svids = sorted(
        int(column)
        for column in dd_aoa.columns
        if int(column) in {int(ephem_column) for ephem_column in ephemeris_aoa.columns}
    )
    if not shared_svids:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.set_title(title)
        ax.text(0.5, 0.5, "No shared SVIDs", ha="center", va="center")
        ax.axis("off")
        return fig

    ncols = min(3, len(shared_svids))
    nrows = int(np.ceil(len(shared_svids) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.8 * ncols, 3.2 * nrows),
        sharex=True,
        sharey=True,
    )
    axes_array = np.atleast_1d(axes).ravel()
    highlight_set = set(int(svid) for svid in highlight_svids)

    for ax, svid in zip(axes_array, shared_svids):
        joined = pd.concat(
            [
                dd_aoa[svid].rename("dd_aoa_deg"),
                ephemeris_aoa[svid].rename("ephemeris_aoa_deg"),
            ],
            axis=1,
        )
        if joined["dd_aoa_deg"].notna().any():
            ax.plot(
                joined.index,
                joined["dd_aoa_deg"],
                color="#2c7fb8",
                linewidth=1.4,
                label="DD",
            )
        if joined["ephemeris_aoa_deg"].notna().any():
            ax.plot(
                joined.index,
                joined["ephemeris_aoa_deg"],
                color="black",
                linestyle="--",
                linewidth=1.3,
                label="ephemeris",
            )
        comparison = joined.dropna()
        error_label = ""
        if not comparison.empty:
            median_error = (
                comparison["dd_aoa_deg"] - comparison["ephemeris_aoa_deg"]
            ).abs().median()
            error_label = f", med err {median_error:.1f} deg"
        suffix = " spoof" if svid in highlight_set else ""
        ax.set_title(f"SV {svid}{suffix}{error_label}")
        ax.set_ylim(0, 180)
        ax.grid(True, alpha=0.3)

    for ax in axes_array[len(shared_svids) :]:
        ax.axis("off")
    axes_array[0].legend(fontsize=8)
    fig.suptitle(title)
    fig.supxlabel("Epoch index")
    fig.supylabel("AoA (deg)")
    fig.tight_layout()
    return fig


def plot_selected_aoa_tracks(result: AoaRunResult):
    return _plot_wide_dataframe(
        result.selected_aoa_deg,
        "Selected DD-method AoA Tracks",
        "Selected AoA (deg)",
        ylim=(0, 180),
    )


def plot_collapse_metrics(result: AoaRunResult):
    plt = _import_pyplot()
    fig, ax = plt.subplots(figsize=(12, 4))
    data = result.collapse_metrics_by_epoch
    ax.plot(
        data.index,
        data["max_minus_min_dispersion_cycles"],
        label="max-min dispersion",
        linewidth=1.4,
    )
    ax.plot(
        data.index,
        data["median_absolute_deviation_cycles"],
        label="median absolute deviation",
        linewidth=1.2,
    )
    ax.set_title("Cross-satellite Relative DD Collapse Metrics")
    ax.set_xlabel("Epoch index")
    ax.set_ylabel("Cycles")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_aoa_collapse_metrics(result: AoaRunResult):
    plt = _import_pyplot()
    fig, ax = plt.subplots(figsize=(12, 4))
    data = result.aoa_collapse_metrics_by_epoch
    if not data.empty:
        ax.plot(
            data.index,
            data["max_minus_min_dispersion_deg"],
            label="max-min dispersion",
            linewidth=1.4,
        )
        ax.plot(
            data.index,
            data["median_absolute_deviation_deg"],
            label="median absolute deviation",
            linewidth=1.2,
        )
    ax.set_title("Cross-satellite AoA Collapse Metrics")
    ax.set_xlabel("Epoch index")
    ax.set_ylabel("Degrees")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_navsat_aoa_diagnostic(result: AoaRunResult):
    return _plot_wide_dataframe(
        result.navsat_aoa_diagnostic_deg,
        "NAV-SAT AoA Diagnostic",
        "Diagnostic AoA (deg)",
        ylim=(0, 180),
    )


def plot_aoa_tracks(
    aoa_tracks: pd.DataFrame,
    *,
    title: str,
    highlight_svids: Sequence[int] = (),
):
    """Plot baseline-relative AoA tracks, emphasizing selected SVIDs."""
    plt = _import_pyplot()
    columns = [int(column) for column in aoa_tracks.columns]
    highlight = set(_normalise_svids(highlight_svids) or ())
    colors = plt.cm.tab20(np.linspace(0, 1, max(len(columns), 1)))

    fig, ax = plt.subplots(figsize=(12, 5))
    for color, svid in zip(colors, columns):
        series = aoa_tracks[svid]
        is_highlighted = svid in highlight
        ax.plot(
            series.index,
            series,
            marker="o" if is_highlighted else None,
            markersize=3 if is_highlighted else 0,
            linewidth=2.4 if is_highlighted else 1.0,
            alpha=1.0 if is_highlighted else 0.45,
            color=color,
            label=f"SV {svid}",
        )
    ax.set_title(title)
    ax.set_xlabel("Epoch index")
    ax.set_ylabel("Baseline-relative AoA (deg)")
    ax.set_ylim(0, 180)
    ax.grid(True, alpha=0.3)
    if columns:
        ax.legend(ncol=6, fontsize=8)
    fig.tight_layout()
    return fig


def plot_sv_used_timeline(
    sv_used_by_receiver: dict[str, pd.DataFrame],
    *,
    title: str = "u-blox NAV-SAT svUsed by Receiver",
    highlight_svids: Sequence[int] = (),
):
    """Plot receiver-native svUsed timelines for every NAV-SAT epoch."""
    from matplotlib.colors import BoundaryNorm, ListedColormap

    plt = _import_pyplot()
    all_svids = sorted(
        {
            int(column)
            for frame in sv_used_by_receiver.values()
            for column in frame.columns
        }
        | set(_normalise_svids(highlight_svids) or ())
    )
    highlight = set(_normalise_svids(highlight_svids) or ())
    if not all_svids:
        fig, ax = plt.subplots(figsize=(12, 3))
        ax.set_title(title)
        ax.text(0.5, 0.5, "No NAV-SAT svUsed data", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        return fig

    receiver_count = max(len(sv_used_by_receiver), 1)
    fig_height = min(
        10.0,
        max(6.0, 2.8 * receiver_count + 0.12 * len(all_svids) * receiver_count),
    )
    fig, axes = plt.subplots(
        receiver_count,
        1,
        figsize=(12, fig_height),
        sharex=False,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    cmap = ListedColormap(["#d8d8d8", "#f2b447", "#277f6a"])
    norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap.N)
    image = None

    for ax, (receiver, frame) in zip(axes, sv_used_by_receiver.items()):
        matrix = frame.reindex(columns=all_svids).T
        values = matrix.where(matrix.notna(), -1).astype(float).clip(lower=-1, upper=1)
        image = ax.imshow(
            values.to_numpy(),
            aspect="auto",
            interpolation="nearest",
            cmap=cmap,
            norm=norm,
        )
        ax.set_title(receiver)
        ax.set_ylabel("SVID")
        ax.set_yticks(range(len(all_svids)))
        ax.set_yticklabels([str(svid) for svid in all_svids])
        for tick, svid in zip(ax.get_yticklabels(), all_svids):
            if svid in highlight:
                tick.set_color("crimson")
                tick.set_fontweight("bold")
        ax.set_xlabel("NAV-SAT epoch index")
        ax.grid(False)

    fig.suptitle(title, y=0.99)
    if image is not None:
        cbar = fig.colorbar(
            image, ax=axes.tolist(), ticks=[-1, 0, 1], fraction=0.025, pad=0.02
        )
        cbar.ax.set_yticklabels(["no NAV-SAT", "not used", "used"])
    return fig


def save_artifacts(
    result: AoaRunResult, output_dir: str | Path | None = None
) -> dict[str, Path]:
    """Save CSV artifacts and PNG diagnostics for one AoA run."""
    directory = Path(output_dir) if output_dir is not None else result.config.output_dir
    directory.mkdir(parents=True, exist_ok=True)

    artifacts: dict[str, Path] = {}
    csv_specs = [
        ("epochs.csv", result.epochs, True),
        ("reference_selection.csv", result.reference_selection, False),
        ("carrier_phase_cycles.csv", result.carrier_phase_cycles, True),
        ("single_difference_cycles.csv", result.single_difference_cycles, True),
        ("double_difference_cycles.csv", result.double_difference_cycles, True),
        ("rounded_dd_cycles.csv", result.rounded_dd_cycles, True),
        ("fractional_dd_cycles.csv", result.fractional_dd_cycles, True),
        ("candidate_aoa_deg.csv", result.candidate_aoa_deg, False),
        ("selected_aoa_deg.csv", result.selected_aoa_deg, True),
        ("selected_branch.csv", result.selected_branch, True),
        ("collapse_metrics_by_epoch.csv", result.collapse_metrics_by_epoch, True),
        ("collapse_metrics.csv", result.collapse_metrics, False),
    ]
    if not result.aoa_collapse_metrics.empty:
        csv_specs.extend(
            [
                (
                    "aoa_collapse_metrics_by_epoch.csv",
                    result.aoa_collapse_metrics_by_epoch,
                    True,
                ),
                ("aoa_collapse_metrics.csv", result.aoa_collapse_metrics, False),
            ]
        )
    if not result.navsat_aoa_diagnostic_deg.empty:
        csv_specs.append(
            ("navsat_aoa_diagnostic_deg.csv", result.navsat_aoa_diagnostic_deg, True)
        )

    for filename, frame, include_index in csv_specs:
        path = directory / filename
        _write_csv(frame, path, index=include_index)
        artifacts[filename] = path

    plot_specs = [
        ("carrier_phase_cycles.png", plot_carrier_phase),
        ("single_difference_cycles.png", plot_single_difference),
        ("double_difference_cycles.png", plot_double_difference),
        ("rounded_dd_cycles.png", plot_rounding_steps),
        ("fractional_dd_cycles.png", plot_fractional_dd),
        ("candidate_aoa_deg.png", plot_candidate_cloud),
        ("selected_aoa_deg.png", plot_selected_aoa_tracks),
        ("collapse_metrics.png", plot_collapse_metrics),
    ]
    if not result.aoa_collapse_metrics.empty:
        plot_specs.append(("aoa_collapse_metrics.png", plot_aoa_collapse_metrics))
    if not result.navsat_aoa_diagnostic_deg.empty:
        plot_specs.append(("navsat_aoa_diagnostic_deg.png", plot_navsat_aoa_diagnostic))

    plt = _import_pyplot()
    for filename, plotter in plot_specs:
        fig = plotter(result)
        path = directory / filename
        fig.savefig(path, dpi=180)
        plt.close(fig)
        artifacts[filename] = path

    result.output_paths = artifacts
    return artifacts


def _combine_svids(*groups: Sequence[int] | None) -> tuple[int, ...]:
    values: list[int] = []
    for group in groups:
        values.extend(_normalise_svids(group) or ())
    return tuple(dict.fromkeys(values))


def _paper_scenario_summary(
    *,
    config: DatasetConfig,
    partial_spoof_enabled: bool,
    spoof_svids: tuple[int, ...],
    analysis_result: AoaRunResult,
    all_satellite_result: AoaRunResult,
    reference_alpha_deg: float | None,
    source_epoch_start: int,
    sv_used_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows = [
        {
            "dataset": config.name,
            "partial_spoof_enabled": bool(partial_spoof_enabled),
            "spoof_svids": spoof_svids,
            "excluded_analysis_svids": config.excluded_analysis_svids or (),
            "reference_svid": analysis_result.reference_svid,
            "reference_alpha_deg": reference_alpha_deg,
            "navsat_azimuth_bias_deg": config.navsat_azimuth_bias_deg,
            "source_epoch_start": int(source_epoch_start),
            "analysis_epoch_count": len(analysis_result.epochs),
            "all_satellite_dd_tracks": int(
                all_satellite_result.selected_aoa_deg.notna().any(axis=0).sum()
            ),
            "output_dir": str(analysis_result.config.output_dir),
        }
    ]
    for row in sv_used_summary.itertuples():
        rows[0][f"rx_{row.receiver}_sv{int(row.svid)}_used_epoch_count"] = int(
            row.used_epoch_count
        )
    return pd.DataFrame(rows)


def run_paper_scenario(
    config: DatasetConfig,
    *,
    partial_spoof_enabled: bool = False,
    spoof_svids: Sequence[int] | None = None,
    convergence_threshold_deg: float = 5.0,
    save_outputs: bool = True,
) -> PaperScenarioResult:
    """Run one paper-style AoA flow for all-spoof or partial-spoof scenarios."""
    normalized_spoof_svids = _normalise_svids(spoof_svids) or ()
    output_dir = config.output_dir

    full_obs1 = read_ubx_observations(
        config.receiver1_path,
        config.gnss_id,
        config.sig_id,
    )
    full_obs2 = read_ubx_observations(
        config.receiver2_path,
        config.gnss_id,
        config.sig_id,
    )

    sv_used_rx1 = compute_navsat_sv_used(full_obs1.navsat)
    sv_used_rx2 = compute_navsat_sv_used(full_obs2.navsat)
    summary_svids = normalized_spoof_svids or _combine_svids(
        sv_used_rx1.columns, sv_used_rx2.columns
    )
    sv_used_summary = summarize_sv_used(
        {"Receiver 1": sv_used_rx1, "Receiver 2": sv_used_rx2},
        summary_svids,
    )

    probe_result: AoaRunResult | None = None
    reference_alpha_by_epoch = pd.DataFrame()
    joint_source_aoa = pd.DataFrame()
    joint_source_branch = pd.DataFrame()
    pairwise_convergence = pd.DataFrame()
    cluster_summary = pd.DataFrame()
    reference_alpha_deg: float | None = config.alpha_r_deg
    source_epoch_start = config.start_index

    if partial_spoof_enabled:
        if not normalized_spoof_svids:
            raise ValueError("spoof_svids is required when partial_spoof_enabled=True.")

        excluded_reference_svids = _combine_svids(
            config.excluded_reference_svids,
            normalized_spoof_svids,
        )
        reference_required_svids = _combine_svids(
            config.reference_required_svids,
            normalized_spoof_svids,
        )
        probe_config = replace(
            config,
            alpha_r_deg=None,
            selected_svids=normalized_spoof_svids,
            excluded_reference_svids=excluded_reference_svids,
            reference_required_svids=reference_required_svids,
        )
        probe_result = run_dataset(probe_config)

        valid_spoof_mask = (
            probe_result.fractional_dd_cycles.reindex(
                columns=list(normalized_spoof_svids)
            )
            .notna()
            .all(axis=1)
        )
        epoch_segment = longest_valid_epoch_segment(valid_spoof_mask)
        if not epoch_segment:
            raise ValueError(
                "No continuous epoch segment has all requested spoof satellites valid."
            )

        reference_alpha_by_epoch = compute_reference_alpha_by_epoch(
            probe_result.epochs,
            full_obs1.navsat,
            probe_result.reference_svid,
            baseline_azimuth_bias_deg=config.navsat_azimuth_bias_deg,
            navsat_tow_decimals=config.navpvt_tow_decimals,
            epoch_tow_column="rcvTow_rx1",
        )
        reference_alpha_window = reference_alpha_by_epoch.loc[
            reference_alpha_by_epoch["epoch_index"].isin(epoch_segment)
        ]
        missing_reference_alpha = reference_alpha_window[
            "reference_alpha_deg"
        ].isna()
        if missing_reference_alpha.all():
            raise ValueError(
                "Reference NAV-SAT AoA is unavailable inside the selected "
                "partial-spoof epoch window."
            )
        if config.reference_svid is None and missing_reference_alpha.any():
            raise ValueError(
                "Reference NAV-SAT AoA is missing inside the selected partial-spoof "
                "epoch window."
            )

        reference_alpha_deg = float(
            reference_alpha_window["reference_alpha_deg"].median()
        )
        epoch_lookup = probe_result.epochs.set_index("epoch_index")
        source_epoch_start = int(
            epoch_lookup.loc[epoch_segment[0]].get(
                "source_epoch_index", epoch_segment[0]
            )
        )

        analysis_config = replace(
            probe_config,
            reference_svid=probe_result.reference_svid,
            alpha_r_deg=reference_alpha_deg,
            start_index=source_epoch_start,
            length=len(epoch_segment),
        )
    else:
        analysis_config = config

    analysis_result = run_dataset(analysis_config)
    if save_outputs:
        save_artifacts(analysis_result)

    if partial_spoof_enabled:
        joint_source_aoa, joint_source_branch = select_joint_source_aoa(
            analysis_result.candidate_aoa_deg,
            normalized_spoof_svids,
            return_branches=True,
        )
        pairwise_convergence = compute_pairwise_aoa_convergence(
            joint_source_aoa,
            threshold_deg=convergence_threshold_deg,
        )
        cluster_summary = summarize_converged_aoa_clusters(
            joint_source_aoa,
            threshold_deg=convergence_threshold_deg,
        )
        all_satellite_config = replace(analysis_config, selected_svids=None)
        all_satellite_result = run_dataset(all_satellite_config)
    else:
        all_satellite_result = analysis_result

    all_satellite_dd_aoa = sort_svid_columns(
        all_satellite_result.selected_aoa_deg.dropna(axis=1, how="all")
    )
    all_satellite_svids = tuple(int(column) for column in all_satellite_dd_aoa.columns)
    ephemeris_aoa_rx1 = compute_navsat_aoa_by_synced_epoch(
        all_satellite_result.epochs,
        full_obs1.navsat,
        epoch_tow_column="rcvTow_rx1",
        navsat_tow_decimals=config.navpvt_tow_decimals,
        baseline_azimuth_bias_deg=config.navsat_azimuth_bias_deg,
        selected_svids=all_satellite_svids,
    )
    ephemeris_aoa_rx2 = compute_navsat_aoa_by_synced_epoch(
        all_satellite_result.epochs,
        full_obs2.navsat,
        epoch_tow_column="rcvTow_rx2",
        navsat_tow_decimals=config.navpvt_tow_decimals,
        baseline_azimuth_bias_deg=config.navsat_azimuth_bias_deg,
        selected_svids=all_satellite_svids,
    )
    all_satellite_valid_candidates = valid_m_candidates(
        all_satellite_result.candidate_aoa_deg
    )
    all_satellite_valid_m_summary = summarize_valid_m_candidates(
        all_satellite_result.candidate_aoa_deg,
        reference_aoa=ephemeris_aoa_rx1,
        selected_branch=all_satellite_result.selected_branch,
    )
    ephemeris_aligned_aoa, ephemeris_aligned_branch_series = select_aoa_by_reference(
        all_satellite_result.candidate_aoa_deg,
        ephemeris_aoa_rx1,
        return_branches=True,
    )
    ephemeris_aligned_aoa = sort_svid_columns(
        ephemeris_aligned_aoa.dropna(axis=1, how="all")
    )
    ephemeris_aligned_branch = _branch_series_to_frame(
        ephemeris_aligned_branch_series
    )
    best_m_all_satellite_aoa = build_best_m_all_satellite_aoa(
        all_satellite_dd_aoa,
        ephemeris_aligned_aoa,
        spoof_svids=normalized_spoof_svids,
    )
    aoa_method_equivalence_summary = summarize_aoa_method_equivalence(
        dp_aoa=all_satellite_dd_aoa,
        dp_branch=all_satellite_result.selected_branch,
        ephemeris_aligned_aoa=ephemeris_aligned_aoa,
        ephemeris_aoa=ephemeris_aoa_rx1,
        ephemeris_aligned_branch=ephemeris_aligned_branch_series,
    )

    run_summary = _paper_scenario_summary(
        config=config,
        partial_spoof_enabled=partial_spoof_enabled,
        spoof_svids=normalized_spoof_svids,
        analysis_result=analysis_result,
        all_satellite_result=all_satellite_result,
        reference_alpha_deg=reference_alpha_deg,
        source_epoch_start=source_epoch_start,
        sv_used_summary=sv_used_summary,
    )

    scenario_result = PaperScenarioResult(
        config=config,
        partial_spoof_enabled=partial_spoof_enabled,
        spoof_svids=normalized_spoof_svids,
        probe_result=probe_result,
        analysis_result=analysis_result,
        all_satellite_result=all_satellite_result,
        reference_alpha_by_epoch=reference_alpha_by_epoch,
        joint_source_aoa_deg=joint_source_aoa,
        joint_source_branch=joint_source_branch,
        pairwise_convergence=pairwise_convergence,
        cluster_summary=cluster_summary,
        all_satellite_dd_method_aoa_deg=all_satellite_dd_aoa,
        ephemeris_aoa_rx1_deg=ephemeris_aoa_rx1,
        ephemeris_aoa_rx2_deg=ephemeris_aoa_rx2,
        all_satellite_valid_m_candidates=all_satellite_valid_candidates,
        all_satellite_valid_m_summary=all_satellite_valid_m_summary,
        ephemeris_aligned_dd_method_aoa_deg=ephemeris_aligned_aoa,
        ephemeris_aligned_branch=ephemeris_aligned_branch,
        best_m_all_satellite_aoa_deg=best_m_all_satellite_aoa,
        aoa_method_equivalence_summary=aoa_method_equivalence_summary,
        sv_used_rx1=sv_used_rx1,
        sv_used_rx2=sv_used_rx2,
        sv_used_summary=sv_used_summary,
        run_summary=run_summary,
    )
    if save_outputs:
        scenario_result.output_paths = save_paper_scenario_artifacts(
            scenario_result, output_dir=output_dir
        )
    return scenario_result


def save_paper_scenario_artifacts(
    scenario: PaperScenarioResult,
    output_dir: str | Path | None = None,
) -> dict[str, Path]:
    """Save paper-scenario CSVs and figures that are outside the base DD run."""
    directory = Path(output_dir) if output_dir is not None else scenario.config.output_dir
    directory.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, Path] = {}

    csv_specs = [
        ("paper_scenario_run_summary.csv", scenario.run_summary, False),
        (
            "paper_reference_alpha_by_epoch.csv",
            scenario.reference_alpha_by_epoch,
            False,
        ),
        (
            "paper_spoof_joint_source_aoa_deg.csv",
            scenario.joint_source_aoa_deg,
            True,
        ),
        (
            "paper_spoof_joint_source_branch.csv",
            scenario.joint_source_branch,
            True,
        ),
        ("paper_spoof_pairwise_convergence.csv", scenario.pairwise_convergence, False),
        ("paper_spoof_cluster_summary.csv", scenario.cluster_summary, False),
        (
            "paper_all_satellite_dd_method_aoa_deg.csv",
            scenario.all_satellite_dd_method_aoa_deg,
            True,
        ),
        (
            "paper_all_satellite_fractional_dd_cycles.csv",
            sort_svid_columns(
                scenario.all_satellite_result.fractional_dd_cycles.dropna(
                    axis=1, how="all"
                )
            ),
            True,
        ),
        (
            "paper_all_satellite_ephemeris_aoa_rx1_deg.csv",
            scenario.ephemeris_aoa_rx1_deg,
            True,
        ),
        (
            "paper_all_satellite_ephemeris_aoa_rx2_deg.csv",
            scenario.ephemeris_aoa_rx2_deg,
            True,
        ),
        (
            "paper_all_satellite_valid_m_candidates.csv",
            scenario.all_satellite_valid_m_candidates,
            False,
        ),
        (
            "paper_all_satellite_valid_m_summary.csv",
            scenario.all_satellite_valid_m_summary,
            False,
        ),
        (
            "paper_all_satellite_ephemeris_aligned_dd_method_aoa_deg.csv",
            scenario.ephemeris_aligned_dd_method_aoa_deg,
            True,
        ),
        (
            "paper_all_satellite_ephemeris_aligned_branch.csv",
            scenario.ephemeris_aligned_branch,
            False,
        ),
        (
            "paper_all_satellite_dd_method_aoa_best_m_deg.csv",
            scenario.best_m_all_satellite_aoa_deg,
            True,
        ),
        (
            "paper_all_satellite_aoa_method_equivalence_summary.csv",
            scenario.aoa_method_equivalence_summary,
            False,
        ),
        ("paper_receiver1_sv_used_all_epochs.csv", scenario.sv_used_rx1, True),
        ("paper_receiver2_sv_used_all_epochs.csv", scenario.sv_used_rx2, True),
        ("paper_sv_used_summary.csv", scenario.sv_used_summary, False),
    ]
    for filename, frame, include_index in csv_specs:
        if frame.empty and filename != "paper_reference_alpha_by_epoch.csv":
            continue
        path = directory / filename
        _write_csv(frame, path, index=include_index)
        artifacts[filename] = path

    plot_specs = [
        (
            "paper_fractional_dd.png",
            lambda: plot_fractional_dd(scenario.analysis_result),
        ),
        (
            "paper_spoof_joint_source_dd_method_aoa.png",
            lambda: plot_aoa_tracks(
                scenario.joint_source_aoa_deg,
                title="Spoof-satellite joint-source DD-method AoA",
                highlight_svids=scenario.spoof_svids,
            ),
        ),
        (
            "paper_all_satellite_dd_method_aoa.png",
            lambda: plot_aoa_tracks(
                scenario.all_satellite_dd_method_aoa_deg,
                title="All-satellite DD-method AoA",
                highlight_svids=scenario.spoof_svids,
            ),
        ),
        (
            "paper_all_satellite_fractional_dd.png",
            lambda: plot_fractional_dd(
                scenario.all_satellite_result,
                highlight_svids=scenario.spoof_svids,
            ),
        ),
        (
            "paper_all_satellite_dd_method_aoa_best_m.png",
            lambda: plot_aoa_tracks(
                scenario.best_m_all_satellite_aoa_deg,
                title=(
                    "All-satellite DD-method AoA "
                    "(authentic: best-m vs ephemeris; spoof: DP)"
                ),
                highlight_svids=scenario.spoof_svids,
            ),
        ),
        (
            "paper_all_satellite_ephemeris_aoa_rx1.png",
            lambda: plot_aoa_tracks(
                scenario.ephemeris_aoa_rx1_deg,
                title="All-satellite ephemeris-based AoA (Receiver 1 NAV-SAT)",
                highlight_svids=scenario.spoof_svids,
            ),
        ),
        (
            "paper_all_satellite_ephemeris_aoa_rx2.png",
            lambda: plot_aoa_tracks(
                scenario.ephemeris_aoa_rx2_deg,
                title="All-satellite ephemeris-based AoA (Receiver 2 NAV-SAT)",
                highlight_svids=scenario.spoof_svids,
            ),
        ),
        (
            "paper_all_satellite_valid_m_candidates.png",
            lambda: plot_valid_m_candidates(
                scenario.all_satellite_result.candidate_aoa_deg,
                reference_aoa=scenario.ephemeris_aoa_rx1_deg,
                title="All valid DD AoA candidates by integer branch",
                highlight_svids=scenario.spoof_svids,
            ),
        ),
        (
            "paper_all_satellite_ephemeris_aligned_vs_ephemeris_aoa.png",
            lambda: plot_aoa_method_comparison(
                scenario.ephemeris_aligned_dd_method_aoa_deg,
                scenario.ephemeris_aoa_rx1_deg,
                title="Constant-m DD AoA vs ephemeris AoA",
                highlight_svids=scenario.spoof_svids,
            ),
        ),
        (
            "paper_sv_used_all_epochs_by_receiver.png",
            lambda: plot_sv_used_timeline(
                {"Receiver 1": scenario.sv_used_rx1, "Receiver 2": scenario.sv_used_rx2},
                title="u-blox NAV-SAT svUsed Over All NAV-SAT Epochs",
                highlight_svids=scenario.spoof_svids,
            ),
        ),
    ]
    plt = _import_pyplot()
    for filename, plotter in plot_specs:
        if (
            filename == "paper_spoof_joint_source_dd_method_aoa.png"
            and scenario.joint_source_aoa_deg.empty
        ):
            continue
        fig = plotter()
        path = directory / filename
        fig.savefig(path, dpi=180)
        plt.close(fig)
        artifacts[filename] = path

    scenario.output_paths = artifacts
    return artifacts
