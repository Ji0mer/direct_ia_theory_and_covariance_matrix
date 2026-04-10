#!/usr/bin/env bash

# Source this file from bash:
#   source /home/zepei/research/direct_ia/setup.sh

_direct_ia_return() {
    return "$1" 2>/dev/null || exit "$1"
}

_direct_ia_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_research_root="$(cd "$_direct_ia_root/.." && pwd)"
_conda_sh="/home/zepei/anaconda3/etc/profile.d/conda.sh"
_cosmosis_env="$_research_root/env"
_cosmosis_lib="$_research_root/cosmosis-standard-library"

if [ ! -f "$_conda_sh" ]; then
    echo "direct_ia setup error: missing $_conda_sh" >&2
    _direct_ia_return 1
fi

if [ ! -d "$_cosmosis_env" ]; then
    echo "direct_ia setup error: missing $_cosmosis_env" >&2
    _direct_ia_return 1
fi

if [ ! -d "$_cosmosis_lib" ]; then
    echo "direct_ia setup error: missing $_cosmosis_lib" >&2
    _direct_ia_return 1
fi

source "$_conda_sh"
conda activate "$_cosmosis_env" || _direct_ia_return 1
source cosmosis-configure || _direct_ia_return 1

# Keep CAMB from oversubscribing CPU threads by default.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

export COSMOSIS_LIB="$_cosmosis_lib"
export IA_LIB="$_direct_ia_root"
export DATA_DIR="$IA_LIB/fits_data"

echo "direct_ia environment ready"
echo "COSMOSIS_LIB=$COSMOSIS_LIB"
echo "IA_LIB=$IA_LIB"
echo "DATA_DIR=$DATA_DIR"
