#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: figure_to_eps.sh INPUT [OUTPUT.eps]

Convert a figure to EPS. PDF input is preferred because vector lines and text
remain vector-based. PNG, JPEG, and TIFF inputs are supported but remain raster
images embedded inside the EPS file.

If OUTPUT is omitted, the EPS file is written beside INPUT with the same stem.
EOF
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
    usage >&2
    exit 2
fi

input=$1
if [[ ! -f "$input" ]]; then
    echo "Input figure does not exist: $input" >&2
    exit 1
fi

input_name=$(basename "$input")
input_stem=${input_name%.*}
input_dir=$(dirname "$input")
output=${2:-"$input_dir/$input_stem.eps"}

if [[ "${output##*.}" != "eps" && "${output##*.}" != "EPS" ]]; then
    echo "Output filename must end in .eps: $output" >&2
    exit 2
fi

mkdir -p "$(dirname "$output")"

extension=$(printf '%s' "${input##*.}" | tr '[:upper:]' '[:lower:]')
case "$extension" in
    pdf)
        if ! command -v pdftops >/dev/null 2>&1; then
            echo "pdftops is required for PDF-to-EPS conversion." >&2
            exit 1
        fi
        pdftops -f 1 -l 1 -eps "$input" "$output"
        ;;
    png|jpg|jpeg|tif|tiff)
        if ! command -v magick >/dev/null 2>&1; then
            echo "ImageMagick (magick) is required for raster-to-EPS conversion." >&2
            exit 1
        fi
        magick "$input" "eps3:$output"
        ;;
    *)
        echo "Unsupported input format: .$extension" >&2
        echo "Supported inputs: PDF, PNG, JPEG, TIFF." >&2
        exit 2
        ;;
esac

if [[ ! -s "$output" ]]; then
    echo "EPS conversion did not produce a non-empty file: $output" >&2
    exit 1
fi

echo "$output"
