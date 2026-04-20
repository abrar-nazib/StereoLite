#!/usr/bin/env bash
# Download FlyingThings3D (frames_finalpass + disparity) from Freiburg.
#
# Usage:
#   bash model/scripts/download_flyingthings3d.sh
#
# Writes tarballs to /media/abrar/AbrarSSD/Datasets/sceneflow_flyingthings3d/
# then extracts them and creates a symlink at data/flyingthings3d.
#
# Sizes you are signing up for:
#   flyingthings3d__frames_finalpass.tar       ~57 GB  (RGB images, left+right)
#   flyingthings3d__disparity.tar.bz2          ~11 GB  (disparity PFM)
#   ------------------------------------------
#   total download:                            ~68 GB
#   extracted size:                            ~110 GB  (bz2 expands a lot)
#
# wget uses -c (continue) so if the connection drops, re-run the same
# command and it resumes from the last byte. There's also a retry loop
# at the shell level in case the server closes the connection mid-stream.
set -euo pipefail

TARGET="${TARGET:-/media/abrar/AbrarSSD/Datasets/sceneflow_flyingthings3d}"
LINK="${LINK:-$(cd "$(dirname "$0")/../.." && pwd)/data/flyingthings3d}"

BASE="https://lmb.informatik.uni-freiburg.de/data/SceneFlowDatasets_CVPR16/Release_april16/data/FlyingThings3D"
FRAMES_URL="${BASE}/raw_data/flyingthings3d__frames_finalpass.tar"
DISP_URL="${BASE}/derived_data/flyingthings3d__disparity.tar.bz2"

mkdir -p "${TARGET}"
cd "${TARGET}"

echo "[FT3D] target dir: ${TARGET}"
echo "[FT3D] df -h:"
df -h "${TARGET}" | awk 'NR==1 || NR==2 {print "  " $0}'
echo ""

# --- robust download wrapper ---------------------------------------------- #
dl() {
    local url="$1"
    local fname
    fname="$(basename "${url}")"
    echo "[FT3D] ========================================================"
    echo "[FT3D] downloading ${fname}"
    echo "[FT3D] from        ${url}"
    echo "[FT3D] ========================================================"
    local attempt=1
    local max_attempts=50
    while (( attempt <= max_attempts )); do
        if wget -c \
                --tries=0 \
                --waitretry=30 \
                --timeout=120 \
                --read-timeout=120 \
                --retry-connrefused \
                --show-progress \
                --progress=bar:force:noscroll \
                "${url}"; then
            echo "[FT3D] ${fname} complete."
            return 0
        fi
        echo "[FT3D] attempt ${attempt} failed, sleeping 60s then retrying..."
        sleep 60
        attempt=$(( attempt + 1 ))
    done
    echo "[FT3D] gave up after ${max_attempts} attempts on ${url}" >&2
    return 1
}

# --- extraction helpers --------------------------------------------------- #
extract_tar() {
    local tar="$1"
    local marker="$2"
    if [[ -d "${marker}" ]]; then
        echo "[FT3D] already extracted: ${marker}"
        return 0
    fi
    echo "[FT3D] extracting $(basename "${tar}") ..."
    case "${tar}" in
        *.tar.bz2) tar xjf "${tar}" ;;
        *.tar)     tar xf  "${tar}" ;;
    esac
    echo "[FT3D] extraction done."
}

# --- main flow ------------------------------------------------------------ #
dl "${FRAMES_URL}"
dl "${DISP_URL}"

echo ""
echo "[FT3D] downloads complete. Sizes:"
ls -lh "${TARGET}"/*.tar* 2>/dev/null | awk '{print "  " $5 "  " $9}'
echo ""

# Extract. FlyingThings3D tars include a top-level directory like
# 'frames_finalpass/TRAIN/A/0000/left/0006.png', so we extract into the
# target directory and the top-level folders land as siblings.
extract_tar "${TARGET}/flyingthings3d__frames_finalpass.tar" \
            "${TARGET}/frames_finalpass"
extract_tar "${TARGET}/flyingthings3d__disparity.tar.bz2" \
            "${TARGET}/disparity"

# Keep the tar files? Comment out the rm to keep them as a backup.
# rm -f "${TARGET}"/flyingthings3d__*.tar*

# --- symlink into the project -------------------------------------------- #
mkdir -p "$(dirname "${LINK}")"
if [[ -L "${LINK}" ]] || [[ -e "${LINK}" ]]; then
    echo "[FT3D] symlink target already exists at ${LINK} — leaving alone."
else
    ln -s "${TARGET}" "${LINK}"
    echo "[FT3D] symlinked ${LINK} -> ${TARGET}"
fi

echo ""
echo "[FT3D] all done."
echo "  download/extract dir:  ${TARGET}"
echo "  project symlink:       ${LINK}"
echo "  ls ${LINK}/:"
ls -la "${LINK}/" 2>/dev/null | head -10
