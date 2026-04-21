import os
import numpy as np
from scipy.integrate import simpson
from scipy.interpolate import interp1d
from scipy.special import jn
from concurrent.futures import ProcessPoolExecutor, as_completed

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(MODULE_DIR, "..", ".."))
DEFAULT_AVG_JN_PATH = os.path.join(
    os.environ.get("IA_LIB", REPO_ROOT), "output", "avg_jn"
)

#############################################################################################################################################
#############################################################################################################################################


def ht(nv, k, fk, rout, kres=5e-4):
    interp_fk = interp1d(k, fk)
    knew = np.arange(k[0], k[-1], kres)
    fknew = interp_fk(knew)
    kr = np.outer(rout, knew)
    j = jn(nv, kr)
    Fr = simpson(j * fknew * knew / (2 * np.pi), x=knew)
    return Fr


def iht(nv, r, Fr, kout, rres=5e-4):
    interp_Fr = interp1d(r, Fr)
    rnew = np.arange(r[0], r[-1], rres)
    Frnew = interp_Fr(rnew)
    kr = np.outer(kout, rnew)
    j = jn(nv, kr)
    fk = simpson(j * Frnew * rnew * 2 * np.pi, x=rnew)
    return fk


def interp_func(x, y, xnew, axis=0, kind="linear"):
    interp_func = interp1d(x, y, axis=axis, kind=kind, fill_value="extrapolate")
    y_new = interp_func(xnew)
    return y_new


# Copied from CosmoSIS.
def compute_c1_baseline():
    C1_M_sun = 5e-14  # h^-2 M_S^-1 Mpc^3
    M_sun = 1.9891e30  # kg
    Mpc_in_m = 3.0857e22  # meters
    C1_SI = C1_M_sun / M_sun * (Mpc_in_m) ** 3  # h^-2 kg^-1 m^3
    # rho_crit_0 = 3 H^2 / 8 pi G
    G = 6.67384e-11  # m^3 kg^-1 s^-2
    H = 100  # h km s^-1 Mpc^-1
    H_SI = H * 1000.0 / Mpc_in_m  # h s^-1
    rho_crit_0 = 3 * H_SI**2 / (8 * np.pi * G)  # h^2 kg m^-3
    f = C1_SI * rho_crit_0
    return f


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


#############################################################################################################################################
#############################################################################################################################################


#############################################################################################################################################
#############################################################################################################################################


class Compute_covmat:
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
    ):
        self.rbins = rbins
        self.res = rres
        self.ktemp = kuse
        self.k = kuse
        self.nv = nv
        self.rp = {}
        self.j = {}
        self.avg_jn = avg_jn
        self.quad_limits = quad_limits
        if load_data is False:
            self.set_jn_data()
        elif load_data is True:
            if path is None:
                self.load_jn_data()
            else:
                self.load_jn_data(file_path=path)

    def set_power_and_w(self, omega_d, omega_s, z, chi, wds, wss, wdd, Pi_max=144.93):
        self.z = np.asarray(z, dtype=float)
        self.omega_d = float(omega_d)
        self.omega_s = float(omega_s)
        self.chi = np.asarray(chi, dtype=float)
        self.dchidz = np.gradient(self.chi, self.z)
        self.wds = np.asarray(wds, dtype=float)
        self.wss = np.asarray(wss, dtype=float)
        self.wdd = np.asarray(wdd, dtype=float)
        self.Pi_max = float(Pi_max)
        return 0

    def save_jn_data(self, file_path="./data/avg_jn/simpson/"):
        os.makedirs(file_path, exist_ok=True)
        np.save(os.path.join(file_path, "k.npy"), self.k)
        np.save(os.path.join(file_path, "rbins.npy"), self.rbins)

        np.save(os.path.join(file_path, "rp_nv0.npy"), self.rp[0])
        np.save(os.path.join(file_path, "rp_nv2.npy"), self.rp[2])
        np.save(os.path.join(file_path, "rp_nv04.npy"), self.rp["[0, 4]"])

        for i in range(len(self.j[0])):
            np.save(os.path.join(file_path, "j0_" + str(i) + ".npy"), self.j[0][i])

        for i in range(len(self.j[2])):
            np.save(os.path.join(file_path, "j2_" + str(i) + ".npy"), self.j[2][i])

        for i in range(len(self.j["[0, 4]"])):
            np.save(
                os.path.join(file_path, "j04_" + str(i) + ".npy"),
                self.j["[0, 4]"][i],
            )
        print("Finished...")

    def set_jn_data(self):
        print("Compute Bessel function parallel...")

        if self.avg_jn:
            tasks = []
            keys = []

            for i in self.nv:
                if isinstance(i, list):
                    print(i)
                    tasks.append(("avg_jns", i))
                    keys.append(str(i))
                elif isinstance(i, int):
                    print(i)
                    tasks.append(("avg_jn", i))
                    keys.append(i)
                else:
                    self.rp[i], self.j[i] = 0, 0

            if tasks:
                max_workers = min(len(tasks), os.cpu_count())
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    future_to_key = {}
                    for task, key in zip(tasks, keys):
                        if task[0] == "avg_jns":
                            future = executor.submit(self.compute_avg_jns, task[1])
                        else:
                            future = executor.submit(self.compute_avg_jn, task[1])
                        future_to_key[future] = key

                    for future in as_completed(future_to_key):
                        key = future_to_key[future]
                        try:
                            rp_result, j_result = future.result()
                            self.rp[key] = rp_result
                            self.j[key] = j_result
                        except Exception as exc:
                            print(f"Task {key} generated an exception: {exc}")
                            self.rp[key], self.j[key] = 0, 0
        else:
            for i in self.nv:
                if isinstance(i, int):
                    self.rp[i], self.j[i] = self.compute_jn(i)
                elif isinstance(i, list):
                    key = str(i)
                    self.rp[key], self.j[key] = self.compute_jns(i)
                else:
                    self.rp[i], self.j[i] = 0, 0

        return 0

    def load_jn_data(self, file_path=None, numbins=None):
        if file_path is None:
            file_path = DEFAULT_AVG_JN_PATH
        if numbins is None:
            numbins = len(self.rbins) - 1
        file_path = os.path.abspath(os.path.expanduser(os.path.expandvars(file_path)))
        if not os.path.isdir(file_path):
            raise FileNotFoundError(
                "Missing averaged Bessel cache directory: %s" % file_path
            )

        print("Only using saved k, rp, averaged jn....")

        cached_rbins = np.load(os.path.join(file_path, "rbins.npy"))
        if (cached_rbins.shape != self.rbins.shape) or (
            not np.allclose(cached_rbins, self.rbins)
        ):
            raise ValueError(
                "Saved averaged Bessel cache rbins do not match current rbins."
            )

        self.k = np.load(os.path.join(file_path, "k.npy"))

        self.rp[0] = np.load(os.path.join(file_path, "rp_nv0.npy"))
        self.rp[2] = np.load(os.path.join(file_path, "rp_nv2.npy"))
        self.rp["[0, 4]"] = np.load(os.path.join(file_path, "rp_nv04.npy"))

        j0 = {}
        j2 = {}
        j04 = {}
        for i in range(numbins):
            j0[i] = np.load(os.path.join(file_path, "j0_" + str(i) + ".npy"))
            j2[i] = np.load(os.path.join(file_path, "j2_" + str(i) + ".npy"))
            j04[i] = np.load(os.path.join(file_path, "j04_" + str(i) + ".npy"))
        self.j[0] = j0
        self.j[2] = j2
        self.j["[0, 4]"] = j04

        return True

    def compute_jn(self, nvi):
        j = {}
        rnew = (self.rbins[:-1] + self.rbins[1:]) / 2
        for ind1 in range(len(rnew)):
            kr = self.k * rnew[ind1]
            j[ind1] = jn(nvi, kr)

        return rnew, j

    def compute_jns(self, nvi):
        j = {}
        rnew = (self.rbins[:-1] + self.rbins[1:]) / 2
        for ind1 in range(len(rnew)):
            kr = self.k * rnew[ind1]
            sum_jns = np.zeros_like(kr)
            for ind2 in nvi:
                sum_jns += jn(ind2, kr)
            j[ind1] = sum_jns

        return rnew, j

    def compute_avg_jn(self, nvi):
        avg_j = {}
        for i in range(len(self.rbins) - 1):
            if self.rbins[i + 1] < 1:
                ruse = np.arange(self.rbins[i], self.rbins[i + 1], self.res / 5)
            else:
                ruse = np.arange(self.rbins[i], self.rbins[i + 1], self.res)
            kr = np.outer(self.k, ruse)
            avg_jn = simpson(2 * np.pi * ruse * jn(nvi, kr), x=ruse)
            avg_jn /= np.pi * (np.max(ruse) ** 2 - np.min(ruse) ** 2)
            avg_j[i] = avg_jn

        rnew = (self.rbins[:-1] + self.rbins[1:]) / 2
        return rnew, avg_j

    def compute_avg_jns(self, nvi):
        avg_j = {}
        for i in range(len(self.rbins) - 1):
            if self.rbins[i + 1] < 1:
                ruse = np.arange(self.rbins[i], self.rbins[i + 1], self.res / 5)
            else:
                ruse = np.arange(self.rbins[i], self.rbins[i + 1], self.res)
            kr = np.outer(self.k, ruse)
            sum_jn = np.zeros_like(kr)
            for j in nvi:
                sum_jn += jn(j, kr)
            avg_jn = simpson(2 * np.pi * ruse * sum_jn, x=ruse)
            avg_jn /= simpson(2 * np.pi * ruse, x=ruse)
            avg_j[i] = avg_jn

        rnew = (self.rbins[:-1] + self.rbins[1:]) / 2
        return rnew, avg_j

    def _require_projection_state(self):
        required = [
            "z",
            "chi",
            "dchidz",
            "wds",
            "wss",
            "wdd",
            "omega_d",
            "omega_s",
            "Pi_max",
        ]
        missing = [name for name in required if not hasattr(self, name)]
        if missing:
            raise RuntimeError(
                "Projection weights are not set. Call set_power_and_w first. Missing: %s"
                % ", ".join(missing)
            )

    def _prepare_power(self, power):
        arr = np.asarray(power, dtype=float)
        if arr.ndim == 1:
            if arr.size == self.ktemp.size and self.ktemp.size != self.k.size:
                arr = interp_func(self.ktemp, arr, self.k)
            elif arr.size != self.k.size:
                raise ValueError(
                    "1D power spectrum must have length %d or %d, got %d"
                    % (self.k.size, self.ktemp.size, arr.size)
                )
            arr = np.broadcast_to(arr[None, :], (self.z.size, self.k.size))
        elif arr.ndim == 2:
            if arr.shape == (self.z.size, self.k.size):
                pass
            elif arr.shape == (self.k.size, self.z.size):
                arr = arr.T
            elif arr.shape == (self.z.size, self.ktemp.size):
                arr = interp_func(self.ktemp, arr, self.k, axis=1)
            elif arr.shape == (self.ktemp.size, self.z.size):
                arr = interp_func(self.ktemp, arr.T, self.k, axis=1)
            else:
                raise ValueError(
                    "2D power spectrum must have shape (%d, %d), (%d, %d), (%d, %d), or (%d, %d), got %s"
                    % (
                        self.z.size,
                        self.k.size,
                        self.k.size,
                        self.z.size,
                        self.z.size,
                        self.ktemp.size,
                        self.ktemp.size,
                        self.z.size,
                        arr.shape,
                    )
                )
        else:
            raise ValueError("Power spectrum must be 1D or 2D.")
        return np.asarray(arr, dtype=float)

    def _prepare_noise(self, noise):
        arr = np.asarray(noise, dtype=float)
        if arr.ndim == 0:
            return np.full(self.z.size, float(arr), dtype=float)
        if arr.ndim == 1 and arr.size == self.z.size:
            return arr
        raise ValueError(
            "Noise term must be scalar or 1D with length %d, got shape %s"
            % (self.z.size, arr.shape)
        )

    def _z_prefactor(self, w_left, w_right, omega_s):
        denom = omega_s * self.chi**2 * self.dchidz
        prefactor = np.zeros_like(self.z, dtype=float)
        mask = np.isfinite(denom) & (denom != 0.0)
        prefactor[mask] = (
            2.0 * self.Pi_max * w_left[mask] * w_right[mask] / denom[mask]
        )
        return prefactor

    def _noise_matrix(self, noise):
        if noise is None:
            return 0.0
        return self._prepare_noise(noise)[:, None]

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
        self._require_projection_state()

        p_ag = self._prepare_power(p_ag)
        p_be = self._prepare_power(p_be)
        p_ae = self._prepare_power(p_ae)
        p_bg = self._prepare_power(p_bg)

        n_ag = self._noise_matrix(n_ag)
        n_be = self._noise_matrix(n_be)
        n_ae = self._noise_matrix(n_ae)
        n_bg = self._noise_matrix(n_bg)

        spectral_term = (p_ag + n_ag) * (p_be + n_be) + (p_ae + n_ae) * (p_bg + n_bg)
        z_prefactor = self._z_prefactor(
            np.asarray(w_left, dtype=float),
            np.asarray(w_right, dtype=float),
            float(omega_s),
        )

        nbins = len(self.rbins) - 1
        cov = np.zeros((nbins, nbins), dtype=float)

        for i in range(nbins):
            left_kernel = np.asarray(self.j[left_key][i], dtype=float)
            for j in range(nbins):
                right_kernel = np.asarray(self.j[right_key][j], dtype=float)
                k_integrand = (self.k / (2 * np.pi)) * left_kernel * right_kernel
                z_integrand = simpson(
                    k_integrand[None, :] * spectral_term, x=self.k, axis=1
                )
                cov[i, j] = simpson(z_prefactor * z_integrand, x=self.z)

        return cov

    def covariance_wgpwgp(self, pgg, pii, pgi, Ng=0, Np=0):
        return self._covariance_block(
            2,
            2,
            self.wds,
            self.wds,
            self.omega_s,
            pgg,
            pii,
            pgi,
            pgi,
            n_ag=Ng,
            n_be=Np,
        )

    def covariance_wgpwpp(self, pgg, pii, pgi, Ng=0, Np=0):
        return self._covariance_block(
            2,
            "[0, 4]",
            self.wds,
            self.wss,
            self.omega_s,
            pgi,
            pii,
            pgi,
            pii,
            n_be=Np,
            n_bg=Np,
        )

    def covariance_wgpwgg(self, pgg, pii, pgi, Ng=0, Np=0):
        return self._covariance_block(
            2,
            0,
            self.wds,
            self.wdd,
            self.omega_s,
            pgg,
            pgi,
            pgg,
            pgi,
            n_ag=Ng,
            n_ae=Ng,
        )

    def covariance_wggwgg(self, pgg, Ng=0):
        return self._covariance_block(
            0,
            0,
            self.wdd,
            self.wdd,
            self.omega_d,
            pgg,
            pgg,
            pgg,
            pgg,
            n_ag=Ng,
            n_be=Ng,
            n_ae=Ng,
            n_bg=Ng,
        )

    def covariance_wggwgp(self, pgg, pii, pgi, Ng=0, Np=0):
        return self._covariance_block(
            0,
            2,
            self.wdd,
            self.wds,
            self.omega_s,
            pgg,
            pgi,
            pgi,
            pgg,
            n_ag=Ng,
            n_bg=Ng,
        )

    def covariance_wggwpp(self, pgg, pii, pgi, Ng=0, Np=0):
        return self._covariance_block(
            0,
            "[0, 4]",
            self.wdd,
            self.wss,
            self.omega_s,
            pgi,
            pgi,
            pgi,
            pgi,
        )

    def covariance_wppwpp(self, pii, Np=0):
        return self._covariance_block(
            "[0, 4]",
            "[0, 4]",
            self.wss,
            self.wss,
            self.omega_s,
            pii,
            pii,
            pii,
            pii,
            n_ag=Np,
            n_be=Np,
            n_ae=Np,
            n_bg=Np,
        )

    def covariance_wppwgp(self, pgg, pii, pgi, Ng=0, Np=0):
        return self._covariance_block(
            "[0, 4]",
            2,
            self.wss,
            self.wds,
            self.omega_s,
            pgi,
            pii,
            pii,
            pgi,
            n_be=Np,
            n_ae=Np,
        )

    def covariance_wppwgg(self, pgg, pii, pgi, Ng=0, Np=0):
        return self._covariance_block(
            "[0, 4]",
            0,
            self.wss,
            self.wdd,
            self.omega_s,
            pgi,
            pgi,
            pgi,
            pgi,
        )
