import numpy as np
from dht_analytic_bessel_nz import Compute_covmat
from scipy.interpolate import interp1d
from cosmosis.datablock import option_section
from astropy.cosmology import Planck13


_TRAPEZOID = getattr(np, "trapezoid", None)
if _TRAPEZOID is None:
    _TRAPEZOID = getattr(np, "trapz", None)
if _TRAPEZOID is None:
    raise ImportError("NumPy must provide trapezoid or trapz integration.")


def _trapz(y, x=None, axis=-1):
    return _TRAPEZOID(y, x=x, axis=axis)


def interp_func(x, y, xnew, axis=0, kind="linear"):
    interp_func = interp1d(x, y, axis=axis, kind=kind, fill_value="extrapolate")
    y_new = interp_func(xnew)
    return y_new


def compute_c1_baseline():
    C1_M_sun = 5e-14  # h^-2 M_S^-1 Mpc^3
    M_sun = 1.9891e30  # kg
    Mpc_in_m = 3.0857e22  # meters
    C1_SI = C1_M_sun / M_sun * (Mpc_in_m) ** 3  # h^-2 kg^-1 m^3
    G = 6.67384e-11  # m^3 kg^-1 s^-2
    H = 100  # h km s^-1 Mpc^-1
    H_SI = H * 1000.0 / Mpc_in_m  # h s^-1
    rho_crit_0 = 3 * H_SI**2 / (8 * np.pi * G)  # h^2 kg m^-3
    return C1_SI * rho_crit_0


def compute_c1(A1, Dz, z_out, z_piv=0, alpha1=0, Omega_m=0.3):
    C1_RHOCRIT = compute_c1_baseline()
    return (
        -1.0
        * A1
        * C1_RHOCRIT
        * Omega_m
        / Dz
        * ((1.0 + z_out) / (1.0 + z_piv)) ** alpha1
    )


def _chi_grid(z, chi=None, cosmo=Planck13.clone(H0=69)):
    if chi is not None:
        return np.asarray(chi, dtype=float)
    return cosmo.comoving_distance(z).value


def get_pz_and_n2d_from_nz(z, nz, area, chi=None, cosmo=Planck13.clone(H0=69)):
    z = np.asarray(z, dtype=float)
    nz = np.asarray(nz, dtype=float)
    chi = _chi_grid(z, chi=chi, cosmo=cosmo)
    dchidz = np.gradient(chi, z)
    geom = chi**2 * dchidz
    shell_counts = area * geom * nz
    n2d = _trapz(shell_counts, z) / area
    if (not np.isfinite(n2d)) or (n2d <= 0.0):
        return np.zeros_like(z, dtype=float), 0.0
    pz = nz * geom / n2d
    norm = _trapz(pz, z)
    if (not np.isfinite(norm)) or (norm <= 0.0):
        return np.zeros_like(z, dtype=float), 0.0
    return pz / norm, n2d


def get_pz_from_nz(z, nz, area, chi=None, cosmo=Planck13.clone(H0=69)):
    pz, _ = get_pz_and_n2d_from_nz(z, nz, area, chi=chi, cosmo=cosmo)
    return pz


def gaussian_val(diff, s):
    return np.exp(-(diff**2) / (2 * s**2))


def _photoz_kernel_matrix(z_true, z_center, chi_true, sz):
    z_true = np.asarray(z_true, dtype=float)
    z_center = np.asarray(z_center, dtype=float)
    chi_true = np.asarray(chi_true, dtype=float)

    kernel = np.zeros((z_center.size, z_true.size), dtype=float)
    sz = float(sz)
    valid = np.isfinite(z_center) & (z_center >= 0.0) & np.isfinite(sz) & (sz > 0.0)
    if np.any(valid):
        sigma = sz * (1.0 + z_center[valid])
        diff = z_true[None, :] - z_center[valid, None]
        kernel[valid, :] = gaussian_val(diff, sigma[:, None])

    norm = _trapz(kernel, x=chi_true, axis=1)
    good = np.isfinite(norm) & (norm > 0.0)
    kernel[good] /= norm[good, None]
    kernel[~good] = 0.0
    return kernel


def apply_photoz_scatter_to_pz(z, pz, chi, sz):
    z = np.asarray(z, dtype=float)
    pz = _normalize_weight(np.asarray(pz, dtype=float), z)
    chi = np.asarray(chi, dtype=float)
    sz = float(sz)
    if (not np.isfinite(sz)) or (sz <= 0.0):
        return pz

    # Match scripts/photoz/w*_photoz.py: Gaussian in redshift, normalized over chi.
    kernel_chi = _photoz_kernel_matrix(z, z, chi, sz)

    pz_per_chi = _trapz(pz[:, None] * kernel_chi, x=z, axis=0)
    pz_scattered = pz_per_chi * np.gradient(chi, z)
    return _normalize_weight(pz_scattered, z)


def _trapz_weights(x):
    x = np.asarray(x, dtype=float)
    weights = np.zeros_like(x)
    if x.size == 1:
        weights[0] = 1.0
        return weights
    dx = np.diff(x)
    weights[0] = 0.5 * dx[0]
    weights[-1] = 0.5 * dx[-1]
    if x.size > 2:
        weights[1:-1] = 0.5 * (dx[:-1] + dx[1:])
    return weights


def photoz_pair_projected_pz(z, pz_a, pz_b, chi, sz, pi_max, n_pi=200):
    z = np.asarray(z, dtype=float)
    pz_a = _normalize_weight(np.asarray(pz_a, dtype=float), z)
    pz_b = _normalize_weight(np.asarray(pz_b, dtype=float), z)
    chi = np.asarray(chi, dtype=float)
    sz = float(sz)

    if (not np.isfinite(sz)) or (sz <= 0.0):
        return _normalize_weight(pz_a * pz_b, z)

    dchidz = np.gradient(chi, z)
    geom = chi**2 * dchidz
    obs_weight = np.divide(pz_a * pz_b, geom, out=np.zeros_like(z), where=geom != 0.0)
    obs_weight = _normalize_weight(obs_weight, z)

    kernel_a = _photoz_kernel_matrix(z, z, chi, sz)
    pi_grid = np.linspace(-float(pi_max), float(pi_max), int(n_pi))
    pi_weights = _trapz_weights(pi_grid)

    pair_per_chi = np.zeros_like(z, dtype=float)
    for pi, pi_weight in zip(pi_grid, pi_weights):
        centers_b = z + pi / dchidz
        kernel_b = _photoz_kernel_matrix(z, centers_b, chi, sz)
        pair_per_chi += pi_weight * _trapz(
            obs_weight[:, None] * kernel_a * kernel_b,
            x=z,
            axis=0,
        )

    pair_per_z = pair_per_chi * dchidz
    return _normalize_weight(pair_per_z, z)


def photoz_los_projection_factor(z, chi, sz_a, sz_b=None, pi_max=144.93, n_pi=200):
    z = np.asarray(z, dtype=float)
    chi = np.asarray(chi, dtype=float)
    sz_a = float(sz_a)
    sz_b = sz_a if sz_b is None else float(sz_b)

    if (
        (not np.isfinite(sz_a))
        or (not np.isfinite(sz_b))
        or sz_a <= 0.0
        or sz_b <= 0.0
    ):
        return np.ones_like(z, dtype=float)

    n_pi = max(int(n_pi), 2)
    pi_grid = np.linspace(-float(pi_max), float(pi_max), n_pi)
    pi_weights = _trapz_weights(pi_grid)

    dchidz = np.gradient(chi, z)
    kernel_a = _photoz_kernel_matrix(z, z, chi, sz_a)

    with np.errstate(divide="ignore", invalid="ignore"):
        inv_chi2 = np.divide(
            1.0,
            chi**2,
            out=np.zeros_like(chi, dtype=float),
            where=chi != 0.0,
        )

    q = np.zeros_like(z, dtype=float)
    for pi, pi_weight in zip(pi_grid, pi_weights):
        centers_b = z + pi / dchidz
        kernel_b = _photoz_kernel_matrix(z, centers_b, chi, sz_b)
        q += pi_weight * _trapz(kernel_a * kernel_b * inv_chi2[None, :], x=chi, axis=1)

    q[~np.isfinite(q)] = 0.0
    # The covariance backend already carries the chi^-2 projection geometry.
    # Normalize the LOS mixing to its spec-z limit so the spectrum factor is
    # dimensionless and tends to unity as s_z -> 0.
    return q * chi**2


def nz_from_pz(z, pz, n2d, chi):
    z = np.asarray(z, dtype=float)
    pz = np.asarray(pz, dtype=float)
    chi = np.asarray(chi, dtype=float)
    geom = chi**2 * np.gradient(chi, z)
    return np.divide(pz * n2d, geom, out=np.zeros_like(pz), where=geom > 0.0)


def _get_covmat_param(block, defaults, name):
    if block.has_value("covmat", name):
        return block["covmat", name]
    return defaults[name]


def _normalize_weight(weight, z):
    norm = _trapz(weight, z)
    if (not np.isfinite(norm)) or (norm == 0.0):
        return np.zeros_like(weight, dtype=float)
    return weight / norm


def _window_from_pz_pair(z, pz_a, pz_b, chi):
    chi = np.asarray(chi, dtype=float)
    geom = chi**2 * np.gradient(chi, z)
    window = np.divide(
        pz_a * pz_b,
        geom,
        out=np.zeros_like(pz_a, dtype=float),
        where=geom != 0.0,
    )
    return _normalize_weight(window, z)


def _safe_inverse_density(nz, factor=1.0):
    nz = np.asarray(nz, dtype=float)
    noise = np.zeros_like(nz, dtype=float)
    mask = np.isfinite(nz) & (nz > 0.0)
    noise[mask] = factor / nz[mask]
    return noise


def _project_photoz_power_spectra(pgg, pii, pgi, Ng, Np, q_dd, q_ss, q_ds):
    pgg_tilde = q_dd[:, None] * (pgg + Ng[:, None])
    pii_tilde = q_ss[:, None] * (pii + Np[:, None])
    pgi_tilde = q_ds[:, None] * pgi
    return pgg_tilde, pii_tilde, pgi_tilde


def _guard_photoz_factor_on_support(q, *nz_arrays):
    q_guarded = np.asarray(q, dtype=float).copy()
    support = np.ones_like(q_guarded, dtype=bool)
    for nz in nz_arrays:
        nz = np.asarray(nz, dtype=float)
        support &= np.isfinite(nz) & (nz > 0.0)
    q_guarded[~support] = 1.0
    return q_guarded


def _get_photoz_sz(block, constant_sz, default_sz):
    if not constant_sz:
        return 0.01
    if block.has_value("photoz_errors", "sz"):
        return block["photoz_errors", "sz"]
    return default_sz


def setup(options):
    sample = options.get_string(option_section, "sample", default="cmass")
    defaults = {
        "zeff": options.get_double(option_section, "zeff", default=0.52),
        "area_shape": options.get_double(option_section, "area_shape", default=5000.0),
        "area_dens": options.get_double(option_section, "area_dens", default=5000.0),
        "rmin": options.get_double(option_section, "rmin", default=0.1),
        "rmax": options.get_double(option_section, "rmax", default=350.0),
        "nr": options.get_int(option_section, "nr", default=21),
        # Match the photo-z correlation modules and avoid the chi -> 0
        # singular point in the LOS projection factor.
        "zmin": options.get_double(option_section, "zmin", default=0.01),
        "sigma_e": options.get_double(option_section, "sigma_e", default=0.25),
        "nbar_shape": options.get_double(option_section, "nbar_shape", default=2e-4),
        "nbar_dens": options.get_double(option_section, "nbar_dens", default=2e-4),
    }
    nk = 10000
    constant_sz = options.get_bool(option_section, "constant_sz", default=True)
    default_sz = options.get_double(option_section, "default_sz", default=0.0)
    n_pi = options.get_int(option_section, "N_pi", default=200)
    pi_mask_max = options.get_double(option_section, "Pi_mask_max", default=-1.0)
    return (
        sample,
        defaults,
        nk,
        constant_sz,
        default_sz,
        n_pi,
        pi_mask_max,
    )


def execute(block, config):
    (
        sample,
        defaults,
        nk,
        constant_sz,
        default_sz,
        n_pi,
        pi_mask_max,
    ) = config
    zeff = _get_covmat_param(block, defaults, "zeff")
    area_shape = _get_covmat_param(block, defaults, "area_shape")
    area_dens = _get_covmat_param(block, defaults, "area_dens")
    rmin = _get_covmat_param(block, defaults, "rmin")
    rmax = _get_covmat_param(block, defaults, "rmax")
    nr = int(_get_covmat_param(block, defaults, "nr"))
    zmin = _get_covmat_param(block, defaults, "zmin")
    sigma_e = _get_covmat_param(block, defaults, "sigma_e")

    h0 = block["cosmological_parameters", "h0"]
    rbins = np.logspace(np.log10(rmin), np.log10(rmax), nr)  # Mpc
    Pimax = block["LOS_bin", "Pi_max"] / h0  # Mpc
    Pi_weight_max = Pimax if pi_mask_max < 0.0 else pi_mask_max / h0

    omega_shape = area_shape * (np.pi / 180) ** 2
    omega_dens = area_dens * (np.pi / 180) ** 2

    A1 = block["intrinsic_alignment_parameters", "A1"]
    b1 = block["bias_%s_density" % sample, "b1E_bin1"]

    plin = block["matter_power_lin", "p_k"] / h0**3
    z = block["matter_power_lin", "z"]
    kh = block["matter_power_lin", "k_h"] * h0

    pnl = block["matter_power_nl", "p_k"] / h0**3
    znl = block["matter_power_nl", "z"]
    khnl = block["matter_power_nl", "k_h"] * h0

    kuse = np.logspace(np.log10(khnl[0]), np.log10(khnl[-1]), nk)

    ind = np.where(kh > 0.03)[0][0]
    Dz = np.sqrt(plin[:, ind] / plin[0, ind])
    Dz_interp = interp1d(z, Dz, bounds_error=False, fill_value="extrapolate")

    zuse = np.linspace(max(float(zmin), 1e-8), 4.0, 401)
    Dz_use = Dz_interp(zuse)
    C1_use = compute_c1(A1, Dz_use, zuse)[:, None]

    pnl_on_z = interp_func(znl, pnl, zuse, axis=0)
    pnl_on_zk = interp_func(khnl, pnl_on_z, kuse, axis=1)

    pgg_nl = b1**2 * pnl_on_zk
    pgi_nl = b1 * C1_use * pnl_on_zk
    pii_nl = C1_use**2 * pnl_on_zk

    z_chi = block["distances", "z"]
    chi = block["distances", "d_m"]
    Chi = interp1d(z_chi, chi, bounds_error=False, fill_value="extrapolate")(zuse)

    zs = block["nz_" + sample + "_shape", "z"]
    nzs_raw = block["nz_" + sample + "_shape", "raw"]
    zd = block["nz_" + sample + "_density", "z"]
    nzd_raw = block["nz_" + sample + "_density", "raw"]

    nzs = interp1d(zs, nzs_raw, bounds_error=False, fill_value=0.0)(zuse)
    nzd = interp1d(zd, nzd_raw, bounds_error=False, fill_value=0.0)(zuse)

    sz = _get_photoz_sz(block, constant_sz, default_sz)
    cosmo = Planck13.clone(H0=h0 * 100)
    pzs = get_pz_from_nz(zuse, nzs, omega_shape, cosmo=cosmo)
    pzd = get_pz_from_nz(zuse, nzd, omega_dens, cosmo=cosmo)

    W_ds = _window_from_pz_pair(zuse, pzd, pzs, Chi)
    W_dd = _window_from_pz_pair(zuse, pzd, pzd, Chi)
    W_ss = _window_from_pz_pair(zuse, pzs, pzs, Chi)

    Ng = _safe_inverse_density(nzd, factor=1.0)
    Np = _safe_inverse_density(nzs, factor=sigma_e**2)

    if sz > 0.0:
        q_photoz = photoz_los_projection_factor(
            zuse, Chi, sz, pi_max=Pi_weight_max, n_pi=n_pi
        )
        q_dd = _guard_photoz_factor_on_support(q_photoz, nzd)
        q_ss = _guard_photoz_factor_on_support(q_photoz, nzs)
        q_ds = _guard_photoz_factor_on_support(q_photoz, nzd, nzs)
        pgg_cov, pii_cov, pgi_cov = _project_photoz_power_spectra(
            pgg_nl,
            pii_nl,
            pgi_nl,
            Ng,
            Np,
            q_dd,
            q_ss,
            q_ds,
        )
        Ng_cov = 0.0
        Np_cov = 0.0
    else:
        pgg_cov = pgg_nl
        pii_cov = pii_nl
        pgi_cov = pgi_nl
        Ng_cov = Ng
        Np_cov = Np

    cc = Compute_covmat(
        rbins,
        1e-3,
        kuse,
        nv=[0, 2, [0, 4]],
        load_data=False,
    )
    cc.set_power_and_w(omega_dens, omega_shape, zuse, Chi, W_ds, W_ss, W_dd, Pimax)

    cov_gpgp = cc.covariance_wgpwgp(pgg_cov, pii_cov, pgi_cov, Ng_cov, Np_cov)
    cov_gppp = cc.covariance_wgpwpp(pgg_cov, pii_cov, pgi_cov, Ng_cov, Np_cov)
    cov_gpgg = cc.covariance_wgpwgg(pgg_cov, pii_cov, pgi_cov, Ng_cov, Np_cov)

    cov_gggp = cc.covariance_wggwgp(pgg_cov, pii_cov, pgi_cov, Ng_cov, Np_cov)
    cov_ggpp = cc.covariance_wggwpp(pgg_cov, pii_cov, pgi_cov, Ng_cov, Np_cov)
    cov_gggg = cc.covariance_wggwgg(pgg_cov, Ng_cov)

    cov_ppgp = cc.covariance_wppwgp(pgg_cov, pii_cov, pgi_cov, Ng_cov, Np_cov)
    cov_pppp = cc.covariance_wppwpp(pii_cov, Np_cov)
    cov_ppgg = cc.covariance_wppwgg(pgg_cov, pii_cov, pgi_cov, Ng_cov, Np_cov)

    clen = len(cc.rp[0])
    Cov = np.zeros((3 * clen, 3 * clen))

    Cov[0 * clen : 1 * clen, 0 * clen : 1 * clen] += cov_gpgp
    Cov[0 * clen : 1 * clen, 1 * clen : 2 * clen] += cov_gppp
    Cov[0 * clen : 1 * clen, 2 * clen : 3 * clen] += cov_gpgg

    Cov[1 * clen : 2 * clen, 0 * clen : 1 * clen] += cov_ppgp
    Cov[1 * clen : 2 * clen, 1 * clen : 2 * clen] += cov_pppp
    Cov[1 * clen : 2 * clen, 2 * clen : 3 * clen] += cov_ppgg

    Cov[2 * clen : 3 * clen, 0 * clen : 1 * clen] += cov_gggp
    Cov[2 * clen : 3 * clen, 1 * clen : 2 * clen] += cov_ggpp
    Cov[2 * clen : 3 * clen, 2 * clen : 3 * clen] += cov_gggg

    block["covmat", "Cov"] = Cov * h0**2
    block["covmat", "rp0"] = cc.rp[0] * h0
    block["covmat", "rp2"] = cc.rp[2] * h0
    block["covmat", "rp04"] = cc.rp["[0, 4]"] * h0

    return 0
