import numpy as np
from dht_analytic_bessel_nz import Compute_covmat
from scipy.interpolate import interp1d
from cosmosis.datablock import option_section
from astropy.cosmology import Planck13


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


def get_pz_from_nz(z, nz, area, cosmo=Planck13.clone(H0=69)):
    z = np.asarray(z, dtype=float)
    nz = np.asarray(nz, dtype=float)
    chi = cosmo.comoving_distance(z).value
    dchidz = np.gradient(chi, z)
    shell_counts = area * chi**2 * dchidz * nz
    n2d = np.trapz(shell_counts, z) / area
    if (not np.isfinite(n2d)) or (n2d <= 0.0):
        return np.zeros_like(z, dtype=float)
    pz = nz / n2d * chi**2 * dchidz
    norm = np.trapz(pz, z)
    if (not np.isfinite(norm)) or (norm <= 0.0):
        return np.zeros_like(z, dtype=float)
    return pz / norm


def _get_covmat_param(block, defaults, name):
    if block.has_value("covmat", name):
        return block["covmat", name]
    return defaults[name]


def _normalize_weight(weight, z):
    norm = np.trapz(weight, z)
    if (not np.isfinite(norm)) or (norm == 0.0):
        return np.zeros_like(weight, dtype=float)
    return weight / norm


def _safe_inverse_density(nz, factor=1.0):
    nz = np.asarray(nz, dtype=float)
    noise = np.zeros_like(nz, dtype=float)
    mask = nz > 0.0
    noise[mask] = factor / nz[mask]
    return noise


def setup(options):
    sample = options.get_string(option_section, "sample", default="cmass")
    defaults = {
        "zeff": options.get_double(option_section, "zeff", default=0.52),
        "area_shape": options.get_double(option_section, "area_shape", default=5000.0),
        "area_dens": options.get_double(option_section, "area_dens", default=5000.0),
        "rmin": options.get_double(option_section, "rmin", default=0.1),
        "rmax": options.get_double(option_section, "rmax", default=350.0),
        "nr": options.get_int(option_section, "nr", default=21),
        "sigma_e": options.get_double(option_section, "sigma_e", default=0.25),
        "nbar_shape": options.get_double(option_section, "nbar_shape", default=2e-4),
        "nbar_dens": options.get_double(option_section, "nbar_dens", default=2e-4),
    }
    nk = 10000
    return sample, defaults, nk


def execute(block, config):
    sample, defaults, nk = config
    zeff = _get_covmat_param(block, defaults, "zeff")
    area_shape = _get_covmat_param(block, defaults, "area_shape")
    area_dens = _get_covmat_param(block, defaults, "area_dens")
    rmin = _get_covmat_param(block, defaults, "rmin")
    rmax = _get_covmat_param(block, defaults, "rmax")
    nr = int(_get_covmat_param(block, defaults, "nr"))
    sigma_e = _get_covmat_param(block, defaults, "sigma_e")

    h0 = block["cosmological_parameters", "h0"]
    rbins = np.logspace(np.log10(rmin), np.log10(rmax), nr) #Mpc
    Pimax = block["LOS_bin", "Pi_max"] / h0  # Mpc

    cosmo = Planck13.clone(H0=h0 * 100)
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

    zuse = np.linspace(1e-8, 4.0, 401)
    Dz_use = Dz_interp(zuse)
    C1_use = compute_c1(A1, Dz_use, zuse)[:, None]

    pnl_on_z = interp_func(znl, pnl, zuse, axis=0)
    pnl_on_zk = interp_func(khnl, pnl_on_z, kuse, axis=1)

    pgg_nl = b1**2 * pnl_on_zk
    pgi_nl = b1 * C1_use * pnl_on_zk
    pii_nl = C1_use**2 * pnl_on_zk

    zs = block["nz_" + sample + "_shape", "z"]
    nzs_raw = block["nz_" + sample + "_shape", "raw"]
    zd = block["nz_" + sample + "_density", "z"]
    nzd_raw = block["nz_" + sample + "_density", "raw"]

    nzs = interp1d(zs, nzs_raw, bounds_error=False, fill_value=0.0)(zuse)
    nzd = interp1d(zd, nzd_raw, bounds_error=False, fill_value=0.0)(zuse)

    pzs = get_pz_from_nz(zuse, nzs, omega_shape, cosmo)
    pzd = get_pz_from_nz(zuse, nzd, omega_dens, cosmo)

    z_chi = block["distances", "z"]
    chi = block["distances", "d_m"]
    Chi = interp1d(z_chi, chi, bounds_error=False, fill_value="extrapolate")(zuse)
    dchidz = np.gradient(Chi, zuse)
    geom = Chi**2 * dchidz

    W_ds = np.divide(pzs * pzd, geom, out=np.zeros_like(zuse), where=geom != 0.0)
    W_dd = np.divide(pzd * pzd, geom, out=np.zeros_like(zuse), where=geom != 0.0)
    W_ss = np.divide(pzs * pzs, geom, out=np.zeros_like(zuse), where=geom != 0.0)

    W_ds = _normalize_weight(W_ds, zuse)
    W_dd = _normalize_weight(W_dd, zuse)
    W_ss = _normalize_weight(W_ss, zuse)
    
    Ng = _safe_inverse_density(nzd, factor=1.0)
    Np = _safe_inverse_density(nzs, factor=sigma_e**2)

    cc = Compute_covmat(
        rbins,
        1e-3,
        kuse,
        nv=[0, 2, [0, 4]],
        load_data=False,
    )
    cc.set_power_and_w(omega_dens, omega_shape, zuse, Chi, W_ds, W_ss, W_dd, Pimax)
    
    cov_gpgp = cc.covariance_wgpwgp(pgg_nl, pii_nl, pgi_nl, Ng, Np)
    cov_gppp = cc.covariance_wgpwpp(pgg_nl, pii_nl, pgi_nl, Ng, Np)
    cov_gpgg = cc.covariance_wgpwgg(pgg_nl, pii_nl, pgi_nl, Ng, Np)

    cov_gggp = cc.covariance_wggwgp(pgg_nl, pii_nl, pgi_nl, Ng, Np)
    cov_ggpp = cc.covariance_wggwpp(pgg_nl, pii_nl, pgi_nl, Ng, Np)
    cov_gggg = cc.covariance_wggwgg(pgg_nl, Ng)

    cov_ppgp = cc.covariance_wppwgp(pgg_nl, pii_nl, pgi_nl, Ng, Np)
    cov_pppp = cc.covariance_wppwpp(pii_nl, Np)
    cov_ppgg = cc.covariance_wppwgg(pgg_nl, pii_nl, pgi_nl, Ng, Np)

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
