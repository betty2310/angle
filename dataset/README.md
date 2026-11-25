# Dataset

The UBX captures are not stored in Git. Download them from:

**TODO: add Google Drive link**

and unpack them here so that each capture folder is a direct child of `dataset/`.

Every capture comes from two u-blox ZED-F9P receivers one L1 wavelength apart. `COM3*.ubx` (or
`raw_data_1.ubx`) is receiver 1 and `COM4*.ubx` (or `raw_data_2.ubx`) is receiver 2. The logs
hold `RXM-RAWX`, `NAV-SAT` and `NAV-PVT` (some also `NAV-SIG`). Spoofing signals were made
with [gps-sdr-sim](https://github.com/betty2310/gps-sdr-sim) and sent over a conducted RF path or
inside a shielded enclosure. Most folders have a `note.txt` with the spoofed PRN group and the
transmitter command.

| Folder | Spoofed PRNs | Notes |
| --- | --- | --- |
| `Authentic/` | none | Authentic sky, static, about 3.4 h (290 MB). `gnss-aoa-dd` ships a 2000 s cut of it |
| `SP-2025-04-24/` | all | Early full-spoof capture |
| `SP-ALL-2026-28-05/` | all | Full-spoof capture, 2026-05-27 |
| `SP-PARITAL-3_17_30-2026-05-06/` | 3, 17, 30 | Partial spoof, X300 transmitter; `ver2`–`ver4` are repeats |
| `sp-partial-19-06/18-06/` | 6, 9, 11, 20 | Mixed-tracking case used by `scripts/manuscript_mixed_case/` |
| `sp-partial-19-06/18-06-01/` | 6, 19, 20, 29 | |
| `sp-partial-19-06/18-06-02/` | 6, 11, 13, 19, 20, 21, 29 | |
| `sp-partial-28-08/v1`, `v2` | 10, 23, 24, 25 | Not usable: receiver 2 has no `RXM-RAWX` |
| `sp-partial-28-08/v3` | 10, 23, 24, 25 | Not usable: spoof PRNs never have valid phase in both receivers at once |
| `sp-partial-28-08/v4` | 10, 23, 24 (25 never tracked) | Transmitter at 30°; result 38.3° |
| `sp-partial-28-08/v5` | 10, 23, 24, 25 | Transmitter at 150°; result about 143° |
| `sp-partial-28-08/v8`, `v9` | 15, 18, 24, 29 | Transmitter at about 90°; use `rx1-rx2` order |
| `sp-partial-16-09/` | 15, 18, 20, 23 | Expected about 40°; `v2/` spoofs 15, 18, 23, 25 |
| `sp-partial-19-09/v1`, `v2` | 10, 15, 18, 25 | `v2` has no position fix |
| `sp-partial-19-09/v3`, `v-last` | 10, 18, 25, 32 | X300 transmitter; `v-last` ends in a cold start after which neither receiver fixes |
| `sp-partial-22-09/` | see `note.txt` | Transmitter at 30°, 1 m away (`v1`) and 3 m away (`v2`, `v3`) |
| `sp-partial-25-09/` | — | Three receiver-pair logs, no note |
| `sp-partial-28-09/30`, `60`, `90` | 10, 23, 28, 32 | Transmitter at 30°, 60°, 90°. A mid-run cold start splits each log into `*-1` and `*-2` (byte ranges and SHA-256 in `note.txt`) |
| `sp-partial-28-09/` top level | 10, 23, 28, 32 | Short preliminary log; only PRN 23 has common valid phase |

`docs/HANDOFF_aoa_paper_refactor.md` has the per-capture configuration and validation table.
