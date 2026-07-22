# Mission: GNSS DD AoA Debugging

## Why
You are learning the DD-method AoA branch logic so you can debug partial-spoof GNSS datasets in this repo, especially when a capture produces DD tracks but the selected AoA or spoof convergence looks wrong.

## Success looks like
- Explain why `m` exists in the DD AoA equation.
- Read `aoa_pipeline.py` outputs and identify which branch was chosen for each SVID and epoch.
- Distinguish a bad branch choice from bad fractional-DD data or missing reference-angle data.

## Constraints
- Lessons should stay tied to this repo's code and datasets.
- Prefer short, practical explanations with code references and checks.

## Out of scope
- Full GNSS RTK ambiguity resolution.
- Full 2D/3D spoofing source localization.
