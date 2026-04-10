from __future__ import print_function

import importlib.util
import warnings
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d


_REPO_ROOT = Path(__file__).resolve().parents[3]
_DIRECT_MODULE_PATH = (
    _REPO_ROOT / "direct_ia" / "projection" / "projected_corrs_legendre" / "legendre_interface.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "direct_ia_projection_projected_corrs_legendre", _DIRECT_MODULE_PATH
)
_DIRECT = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_DIRECT)


names = _DIRECT.names
option_section = _DIRECT.option_section
TRAPEZOID = _DIRECT.TRAPEZOID
CUMULATIVE_TRAPEZOID = _DIRECT.CUMULATIVE_TRAPEZOID
y3fid_cosmology = _DIRECT.y3fid_cosmology

interp_power = _DIRECT.interp_power
add_gg_mag_terms = _DIRECT.add_gg_mag_terms
add_gp_lensmag_terms = _DIRECT.add_gp_lensmag_terms


def get_redshift_kernel(block, i, j, z0, *args):
    if len(args) == 3:
        x, sample_a, sample_b = args
        z_x = block["distances", "z"]
    elif len(args) == 4:
        z_x, x, sample_a, sample_b = args
    else:
        raise TypeError(
            "get_redshift_kernel expects either (block, i, j, z0, x, sample_a, sample_b) "
            "or (block, i, j, z0, z_x, x, sample_a, sample_b)"
        )
    return _DIRECT.get_redshift_kernel(block, i, j, z0, z_x, x, sample_a, sample_b)


class Projected_Corr_RSD(_DIRECT.Projected_Corr_RSD):
    # Keep the fallback threshold aligned with the historical protected module.
    STABILITY_THRESHOLD = 1.0e10
    _logged_fallback_reasons = set()

    def get_xi_noext(self, pk=None, l=(0, 2, 4)):
        xi = {}
        for ell in l:
            ri, xi_ell = self.j[ell](pk, extrap=False)
            xi_interp = interp1d(ri, xi_ell, bounds_error=False, fill_value=0)
            xi[ell] = np.dot((xi_interp(self.rG) * self.L[ell]), self.dpi)
            xi[ell] *= 2.0
        return xi

    @classmethod
    def _log_fallback_once(cls, reason):
        if reason not in cls._logged_fallback_reasons:
            print(reason)
            cls._logged_fallback_reasons.add(reason)

    @staticmethod
    def _xi_is_stable(xi, threshold):
        for value in xi.values():
            if not np.isfinite(value).all():
                return False
            if np.max(np.abs(value)) > threshold:
                return False
        return True

    def _get_xi_wgg_protected(self, pk, l, threshold):
        if not np.isfinite(pk).all():
            self._log_fallback_once("wgg: non-finite P(k) input; retrying with mcfit extrap=False")
            return self.get_xi_noext(pk=pk, l=l)

        if np.any(pk <= 0.0):
            # The mcfit extrapolation path is unstable for these sign-changing tails.
            self._log_fallback_once("wgg: non-positive P(k) tail detected; using mcfit extrap=False")
            return self.get_xi_noext(pk=pk, l=l)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", RuntimeWarning)
            xi = self.get_xi(pk=pk, l=l)

        had_runtime_warning = any(issubclass(w.category, RuntimeWarning) for w in caught)
        if had_runtime_warning or (not self._xi_is_stable(xi, threshold)):
            self._log_fallback_once("wgg: mcfit extrap=True became unstable; retrying with extrap=False")
            return self.get_xi_noext(pk=pk, l=l)

        return xi

    def xi_wgg(self, f=0, bg=0, bg2=None, pk=None, xi=None, l=(0, 2, 4), threshold=None):
        bg1 = bg
        if bg2 is None:
            bg2 = bg
        beta1 = f / bg1
        beta2 = f / bg2

        if threshold is None:
            threshold = self.STABILITY_THRESHOLD
        if xi is None or (not self._xi_is_stable(xi, threshold)):
            xi = self._get_xi_wgg_protected(pk=pk, l=l, threshold=threshold)

        W = np.zeros_like(xi[next(iter(xi.keys()))])
        for ell in l:
            W += (xi[ell].T * self.alpha(ell, beta1, beta2) * bg1 * bg2).T
        return W, xi

    def wgg_calc(self, f=0, bg=0, bg2=None, pk=None, xi=None, l=(0, 2, 4)):
        W, _ = self.xi_wgg(
            f=f,
            bg=bg,
            bg2=bg2,
            pk=pk,
            xi=xi,
            l=l,
            threshold=self.STABILITY_THRESHOLD,
        )
        return W


def setup(options):
    return _DIRECT.setup(options)


def execute(block, config):
    original_projected_corr = _DIRECT.Projected_Corr_RSD
    _DIRECT.Projected_Corr_RSD = Projected_Corr_RSD
    try:
        return _DIRECT.execute(block, config)
    finally:
        _DIRECT.Projected_Corr_RSD = original_projected_corr
