"""Data check for a two-receiver UBX capture before running the DD-method AoA flow.

Reports, per receiver: message inventory (RXM-RAWX / NAV-PVT / NAV-SAT / NAV-SIG),
RAWX epochs and time span, per-PRN RAWX rows, rows with usable carrier phase
(cpValid=1 and halfCyc=1), lock time, C/N0 and NAV-SAT svUsed. Then the
synchronized overlap of the two receivers and, for the spoof group, the epochs
where every spoof PRN has valid phase in both receivers (what the notebook's
partial-spoof flow needs) plus each spoof PRN's valid-phase timeline.

    uv run python probe_dataset.py dataset/sp-partial-28-08/v9 --spoof 15 18 24 29
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd

from aoa_pipeline import (
    _make_ubx_reader,
    build_carrier_phase_table,
    longest_valid_epoch_segment,
    read_ubx_observations,
    sync_receiver_epochs,
)

KEY_MESSAGES = ("RXM-RAWX", "NAV-PVT", "NAV-SAT", "NAV-SIG", "RXM-SFRBX", "CFG-MSG")


def receiver_paths(folder: Path) -> tuple[Path, Path]:
    rx1 = sorted(folder.glob("COM3*.ubx"))
    rx2 = sorted(folder.glob("COM4*.ubx"))
    if not rx1 or not rx2:
        raise FileNotFoundError(f"Expected COM3*.ubx and COM4*.ubx in {folder}")
    return rx1[0], rx2[0]


def message_inventory(path: Path) -> Counter:
    counts: Counter = Counter()
    with path.open("rb") as stream:
        for _, parsed in _make_ubx_reader(stream):
            counts[getattr(parsed, "identity", "?")] += 1
    return counts


def segments(mask: pd.Series, tow: pd.Series) -> list[str]:
    out, start, prev = [], None, None
    for idx, ok in mask.items():
        if ok and start is None:
            start = idx
        if not ok and start is not None:
            out.append((start, prev))
            start = None
        if ok:
            prev = idx
    if start is not None:
        out.append((start, prev))
    return [f"TOW {tow.loc[a]:.0f}-{tow.loc[b]:.0f} ({b - a + 1} ep)" for a, b in out]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", type=Path, help="capture folder holding COM3*.ubx and COM4*.ubx")
    parser.add_argument("--spoof", type=int, nargs="*", default=[], help="spoof PRN group")
    args = parser.parse_args()
    pd.set_option("display.width", 250)
    spoof = tuple(args.spoof)

    rx1, rx2 = receiver_paths(args.folder)
    obs = {}
    for label, path in (("rx1", rx1), ("rx2", rx2)):
        inventory = message_inventory(path)
        print(f"== {label}: {path.name} ({path.stat().st_size / 1024:.0f} KB)  "
              + ", ".join(f"{key}:{inventory.get(key, 0)}" for key in KEY_MESSAGES))
        o = read_ubx_observations(path, 0, 0)
        obs[label] = o
        if o.rawx.empty:
            print("   no RXM-RAWX: this receiver contributes no carrier phase; the DD flow cannot run.")
            continue
        rawx = o.rawx
        print(f"   RAWX epochs={rawx['rcvTow'].nunique()} span={rawx['rcvTow'].max() - rawx['rcvTow'].min():.0f}s "
              f"NAV-PVT rows={len(o.navpvt)} NAV-SAT rows={len(o.navsat)}")
        good = rawx["cp_valid"].fillna(0).astype(int).eq(1) & rawx["half_cycle"].fillna(0).astype(int).eq(1)
        per = rawx.assign(good=good).groupby("svid").agg(
            rawx_rows=("cpMes", "size"), usable_phase_rows=("good", "sum"),
            cno_mean=("cno", "mean"), locktime_max_s=("locktime", lambda s: s.max() / 1000.0))
        if not o.navsat.empty:
            nav = o.navsat.groupby("svid").agg(sv_used=("svUsed", "sum"), elev=("elev", "median"), azim=("azim", "median"))
            per = per.join(nav, how="left")
        per["spoof"] = per.index.isin(spoof)
        print(per.round(1).to_string())

    if obs["rx1"].rawx.empty or obs["rx2"].rawx.empty:
        return
    synced = sync_receiver_epochs(obs["rx1"], obs["rx2"], tow_round_decimals=0)
    tables = build_carrier_phase_table(synced, use_navpvt_time_correction=True, navpvt_tow_decimals=0)
    common = tables.rx1_cycles.notna() & tables.rx2_cycles.notna()
    tow = synced.epochs.set_index("epoch_index")["rcvTow_rx1"]
    print(f"== synchronized epochs: {len(synced.epochs)}; common valid-phase epochs per PRN:")
    print(common.sum().to_string())
    if not spoof:
        return
    present = [s for s in spoof if s in common.columns and common[s].any()]
    missing = [s for s in spoof if s not in present]
    print(f"== spoof PRNs with common valid phase: {present}; without: {missing}")
    for s in present:
        both = common[s]
        rx1_only = tables.rx1_cycles[s].notna()
        rx2_only = tables.rx2_cycles[s].notna()
        print(f"   PRN {s}: rx1 {segments(rx1_only, tow)}")
        print(f"   PRN {s}: rx2 {segments(rx2_only, tow)}")
        print(f"   PRN {s}: both {segments(both, tow)}")
    if present:
        all_valid = common.reindex(columns=present).all(axis=1)
        run = longest_valid_epoch_segment(all_valid)
        print(f"== epochs with all present spoof PRNs valid in both receivers: {int(all_valid.sum())}; "
              f"longest continuous run: {len(run)}"
              + (f" (TOW {tow.loc[run[0]]:.0f}-{tow.loc[run[-1]]:.0f})" if run else ""))


if __name__ == "__main__":
    main()
