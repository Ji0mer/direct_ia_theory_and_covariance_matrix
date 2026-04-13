#!/usr/bin/env bash

# Source this file from bash:
#   source /home/jiomer/research/direct_ia_theory_and_covariance_matrix/setup.sh

_direct_ia_return() {
    return "$1" 2>/dev/null || exit "$1"
}

_direct_ia_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_research_root="$(cd "$_direct_ia_root/.." && pwd)"
_home_dir="${HOME:-/home/jiomer}"
_conda_sh="${CONDA_SH:-$_home_dir/anaconda3/etc/profile.d/conda.sh}"
_cosmosis_root="${COSMOSIS_ROOT:-$_research_root/cosmosis}"
_cosmosis_env="${COSMOSIS_ENV:-$_cosmosis_root/env}"
_cosmosis_lib="${COSMOSIS_LIB:-$_cosmosis_root/cosmosis-standard-library}"

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
source "$_cosmosis_env/bin/cosmosis-configure" || _direct_ia_return 1

# Keep CAMB from oversubscribing CPU threads by default.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

export COSMOSIS_LIB="$_cosmosis_lib"
export IA_LIB="$_direct_ia_root"
export DATA_DIR="$IA_LIB/fits_data"

printf 'direct_ia environment ready\n'
printf 'COSMOSIS_LIB=%s\n' "$COSMOSIS_LIB"
printf 'IA_LIB=%s\n' "$IA_LIB"
printf 'DATA_DIR=%s\n' "$DATA_DIR"
