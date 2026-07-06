import numpy as np
from scipy.special import jv

from dht_simpson_nz import Compute_covmat as _SimpsonComputeCovmat


def _simpson_weights_1d(x):
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("Simpson weights require a 1D grid.")

    n = x.size
    if n < 2:
        raise ValueError("Simpson weights require at least two samples.")
    if n == 2:
        dx = x[1] - x[0]
        return np.array([0.5 * dx, 0.5 * dx], dtype=float)

    weights = np.zeros(n, dtype=float)
    stop = n - 3 if n % 2 == 0 else n - 2

    for start in range(0, stop, 2):
        h0 = x[start + 1] - x[start]
        h1 = x[start + 2] - x[start + 1]
        hsum = h0 + h1
        hprod = h0 * h1
        h0divh1 = h0 / h1

        weights[start] += hsum / 6.0 * (2.0 - 1.0 / h0divh1)
        weights[start + 1] += hsum / 6.0 * (hsum * hsum / hprod)
        weights[start + 2] += hsum / 6.0 * (2.0 - h0divh1)

    if n % 2 == 0:
        h0 = x[-2] - x[-3]
        h1 = x[-1] - x[-2]
        alpha = (2.0 * h1**2 + 3.0 * h0 * h1) / (6.0 * (h1 + h0))
        beta = (h1**2 + 3.0 * h0 * h1) / (6.0 * h0)
        eta = h1**3 / (6.0 * h0 * (h0 + h1))
        weights[-1] += alpha
        weights[-2] += beta
        weights[-3] -= eta

    return weights


def _annulus_moment(a, b, order):
    return (b ** (2 * order + 2) - a ** (2 * order + 2)) / (
        (order + 1) * (b**2 - a**2)
    )


def _small_j0(k, a, b):
    k2 = k**2
    return (
        1.0
        - k2 * _annulus_moment(a, b, 1) / 4.0
        + k2**2 * _annulus_moment(a, b, 2) / 64.0
        - k2**3 * _annulus_moment(a, b, 3) / 2304.0
    )


def _small_j2(k, a, b):
    k2 = k**2
    return (
        k2 * _annulus_moment(a, b, 1) / 8.0
        - k2**2 * _annulus_moment(a, b, 2) / 96.0
        + k2**3 * _annulus_moment(a, b, 3) / 3072.0
    )


def _small_j04(k, a, b):
    k2 = k**2
    return (
        1.0
        - k2 * _annulus_moment(a, b, 1) / 4.0
        + 7.0 * k2**2 * _annulus_moment(a, b, 2) / 384.0
        - 13.0 * k2**3 * _annulus_moment(a, b, 3) / 23040.0
    )


def _f2(x):
    return -2.0 * jv(0, x) - x * jv(1, x)


def _f04(x):
    out = np.empty_like(x, dtype=float)
    small = np.abs(x) < 1e-4
    xs = x[small]
    out[small] = (
        xs**2 / 2.0
        - xs**4 / 16.0
        + 7.0 * xs**6 / 2304.0
        - 13.0 * xs**8 / 184320.0
    )
    xl = x[~small]
    out[~small] = 4.0 + (2.0 * (xl**2 - 4.0) / xl) * jv(1, xl) - 8.0 * jv(
        2, xl
    )
    return out


def _grid_inputs(rbins, k):
    rbins = np.asarray(rbins, dtype=float)
    k = np.asarray(k, dtype=float)
    if rbins.ndim != 1 or rbins.size < 2:
        raise ValueError("rbins must be a 1D array with at least two edges.")
    if k.ndim != 1:
        raise ValueError("k must be a 1D array.")

    a = rbins[:-1, None]
    b = rbins[1:, None]
    kval = k[None, :]
    delta = b**2 - a**2
    if np.any(delta <= 0):
        raise ValueError("rbins must be strictly increasing.")
    return a, b, kval, delta


def averaged_j0(rbins, k, small_x=1e-3):
    a, b, kval, delta = _grid_inputs(rbins, k)
    xa = kval * a
    xb = kval * b
    small = np.abs(xb) < small_x

    with np.errstate(divide="ignore", invalid="ignore"):
        out = 2.0 * (b * jv(1, xb) - a * jv(1, xa)) / (kval * delta)
    out[small] = _small_j0(kval, a, b)[small]
    return out


def averaged_j2(rbins, k, small_x=1e-3):
    a, b, kval, delta = _grid_inputs(rbins, k)
    xa = kval * a
    xb = kval * b
    small = np.abs(xb) < small_x

    with np.errstate(divide="ignore", invalid="ignore"):
        out = 2.0 * (_f2(xb) - _f2(xa)) / (kval**2 * delta)
    out[small] = _small_j2(kval, a, b)[small]
    return out


def averaged_j04(rbins, k, small_x=1e-3):
    a, b, kval, delta = _grid_inputs(rbins, k)
    xa = kval * a
    xb = kval * b
    small = np.abs(xb) < small_x

    with np.errstate(divide="ignore", invalid="ignore"):
        out = 2.0 * (_f04(xb) - _f04(xa)) / (kval**2 * delta)
    out[small] = _small_j04(kval, a, b)[small]
    return out


def averaged_bessel_kernels(rbins, k, nv=(0, 2, [0, 4]), small_x=1e-3):
    kernels = {}
    for item in nv:
        if item == 0:
            kernels[0] = averaged_j0(rbins, k, small_x=small_x)
        elif item == 2:
            kernels[2] = averaged_j2(rbins, k, small_x=small_x)
        elif isinstance(item, list) and item == [0, 4]:
            kernels[str(item)] = averaged_j04(rbins, k, small_x=small_x)
        else:
            raise ValueError(
                "Analytical annulus averages are implemented only for 0, 2, and [0, 4]."
            )
    return kernels


class Compute_covmat(_SimpsonComputeCovmat):
    def __init__(
        self,
        rbins,
        rres,
        kuse,
        nv=[0, 2, [0, 4]],
        logspace=True,
        avg_jn=True,
        load_data=False,
        path=None,
        quad_limits=5000,
        small_x=1e-3,
    ):
        if load_data:
            raise ValueError(
                "Analytical Bessel mode computes kernels directly; do not use load_data=True."
            )
        self.small_x = small_x
        super().__init__(
            rbins,
            rres,
            kuse,
            nv=nv,
            logspace=logspace,
            avg_jn=avg_jn,
            load_data=False,
            path=path,
            quad_limits=quad_limits,
        )

    def set_jn_data(self):
        if not self.avg_jn:
            return super().set_jn_data()

        print("Compute analytical averaged Bessel functions...")
        rnew = (self.rbins[:-1] + self.rbins[1:]) / 2.0
        kernels = averaged_bessel_kernels(
            self.rbins, self.k, nv=self.nv, small_x=self.small_x
        )

        for key, values in kernels.items():
            self.rp[key] = rnew
            self.j[key] = {i: values[i].copy() for i in range(values.shape[0])}
        return 0

    def _get_simpson_weights(self):
        cache_key = (
            self.k.shape,
            self.z.shape,
            float(self.k[0]),
            float(self.k[-1]),
            float(self.z[0]),
            float(self.z[-1]),
        )
        if getattr(self, "_simpson_weight_cache_key", None) != cache_key:
            self._simpson_weight_cache_key = cache_key
            self._simpson_k_weights = _simpson_weights_1d(self.k)
            self._simpson_z_weights = _simpson_weights_1d(self.z)
        return self._simpson_k_weights, self._simpson_z_weights

    def _kernel_array(self, key):
        nbins = len(self.rbins) - 1
        return np.vstack(
            [np.asarray(self.j[key][i], dtype=float) for i in range(nbins)]
        )

    def _fast_covariance_from_spectral_term(
        self,
        left_key,
        right_key,
        w_left,
        w_right,
        omega_s,
        spectral_term,
    ):
        self._require_projection_state()

        spectral_term = np.asarray(spectral_term, dtype=float)
        target_shape = (self.z.size, self.k.size)
        if spectral_term.shape != target_shape:
            try:
                spectral_term = np.broadcast_to(spectral_term, target_shape)
            except ValueError as exc:
                raise ValueError(
                    "Spectral term must have shape (%d, %d), got %s"
                    % (self.z.size, self.k.size, spectral_term.shape)
                ) from exc
        if spectral_term.shape != target_shape:
            raise ValueError(
                "Spectral term must have shape (%d, %d), got %s"
                % (self.z.size, self.k.size, spectral_term.shape)
            )

        z_prefactor = self._z_prefactor(
            np.asarray(w_left, dtype=float),
            np.asarray(w_right, dtype=float),
            float(omega_s),
        )
        k_weights, z_weights = self._get_simpson_weights()
        weighted_spectrum = (z_weights * z_prefactor) @ spectral_term
        k_weight = k_weights * self.k / (2.0 * np.pi) * weighted_spectrum

        left_kernel = self._kernel_array(left_key)
        right_kernel = self._kernel_array(right_key)
        return (left_kernel * k_weight[None, :]) @ right_kernel.T

    def _covariance_block(
        self,
        left_key,
        right_key,
        w_left,
        w_right,
        omega_s,
        p_ag,
        p_be,
        p_ae,
        p_bg,
        n_ag=None,
        n_be=None,
        n_ae=None,
        n_bg=None,
    ):
        p_ag = self._prepare_power(p_ag)
        p_be = self._prepare_power(p_be)
        p_ae = self._prepare_power(p_ae)
        p_bg = self._prepare_power(p_bg)

        n_ag = self._noise_matrix(n_ag)
        n_be = self._noise_matrix(n_be)
        n_ae = self._noise_matrix(n_ae)
        n_bg = self._noise_matrix(n_bg)

        spectral_term = (p_ag + n_ag) * (p_be + n_be) + (p_ae + n_ae) * (
            p_bg + n_bg
        )
        return self._fast_covariance_from_spectral_term(
            left_key,
            right_key,
            w_left,
            w_right,
            omega_s,
            spectral_term,
        )

    def _covariance_from_spectral_term(
        self,
        left_key,
        right_key,
        w_left,
        w_right,
        omega_s,
        spectral_term,
    ):
        return self._fast_covariance_from_spectral_term(
            left_key,
            right_key,
            w_left,
            w_right,
            omega_s,
            spectral_term,
        )

    def load_jn_data(self, *args, **kwargs):
        raise RuntimeError("Analytical Bessel mode does not read averaged-Bessel caches.")

    def save_jn_data(self, *args, **kwargs):
        raise RuntimeError("Analytical Bessel mode does not write averaged-Bessel caches.")
