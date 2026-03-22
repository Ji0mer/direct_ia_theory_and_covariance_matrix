from __future__ import print_function
from cosmosis.datablock import names, option_section
import sys
import numpy as np
from scipy.interpolate import interp1d, RectBivariateSpline
from scipy.integrate import quad
from hankl import FFTLog

# constants
clight = 299792.4580 # kms^-1

def setup(options):
    sample_a = options.get_string(option_section, "sample_a", default="forecast_sample_density")
    sample_b = options.get_string(option_section, "sample_b", default="forecast_sample_density")
    timing = options.get_bool(option_section, "timing", default=True)
    constant_sigmaz = options.get_bool(option_section, "constant_sigmaz", default=True)
    n_pi = options.get_int(option_section, "N_pi", default=200)
    pi_mask_max = options.get_double(option_section, "Pi_mask_max", default=-1.0)
    return sample_a, sample_b, timing, constant_sigmaz, n_pi, pi_mask_max

def execute(block, config):
    sample_a, sample_b, timing, constant_sigmaz, n_pi, pi_mask_max = config
    
    if timing:
        from time import time
        T0 = time()
    
    # --- 1. Setup Grids ---
    Npi = n_pi
    Nz = 200
    Pi = np.linspace(-block['LOS_bin','Pi_max'], block['LOS_bin','Pi_max'], Npi)
    z_low = np.linspace(0.01, 4.00, Nz)
    
    z_distance = block["distances","z"]
    chi_distance = block["distances","d_m"] * block['cosmological_parameters', 'h0']
    chi_of_z_spline = interp1d(z_distance, chi_distance, bounds_error=False, fill_value="extrapolate")
    
    zf = np.linspace(0.0, 4.0, 400)
    chi = chi_of_z_spline(zf)
    
    # --- 2. Photo-z parameters ---
    if not constant_sigmaz:
        sigmaz_a = 0.01
        sigmaz_b = 0.01
    else:
        sigmaz_a = block['photoz_errors','sigmaz']
        sigmaz_b = block['photoz_errors','sigmaz']
        
    # --- 3. Power Spectrum Interpolation ---
    # Apply bias terms
    P_gg = block['galaxy_power','p_k'] * block["galaxy_power","blin_1"] * block["galaxy_power","blin_2"]
    k_power = block['galaxy_power','k_h']
    z_power = block['galaxy_power','z']
    chi_power = chi_of_z_spline(z_power)
    
    # Use RectBivariateSpline in Linear Space (Robust against zeros/negatives)
    # Transpose P_gg to match (k, chi)
    P_gg_spline = RectBivariateSpline(np.log(k_power), chi_power, P_gg.T)

    Nell = 300
    ell = np.logspace(-6, np.log10(20000), Nell) 
    
    # Pre-compute P_gg_2d
    chi_safe = chi.copy()
    chi_safe[chi_safe == 0] = 1.0 
    
    ell_grid = ell[:, None]
    k_eval = (ell_grid + 0.5) / chi_safe[None, :]
    
    # Evaluate spline
    P_gg_2d_flat = P_gg_spline.ev(np.log(k_eval), np.tile(chi, (Nell, 1)))
    P_gg_2d = P_gg_2d_flat.reshape(Nell, len(chi))
    
    if timing:
        T1 = time()
        print('Setup done. Starting Super-Fast Integral...')
        
    # --- 4. Super-Fast Limber Integral ---
    # Strategy: Integrate Kernel over Pi FIRST, then multiply P(k).
    
    # A. Geometry
    Hz_all = 100 * np.sqrt(block['cosmological_parameters','omega_m']*(1+z_low)**3 + block['cosmological_parameters', 'omega_lambda'])
    z1_grid = z_low[:, None] 
    z2_grid = z_low[:, None] + (1./clight * Hz_all[:, None] * Pi[None, :]) 
    
    # B. Gaussian Weights (Nz, Npi, 400)
    zf_b = zf[None, None, :]
    
    # Pz1
    diff1 = zf_b - z1_grid[:, :, None]
    Pz1_mat = gaussian_val(diff1, sigmaz_a)
    norm1 = np.trapz(Pz1_mat, x=chi, axis=-1)
    norm1[norm1 == 0] = 1.0
    Pz1_mat /= norm1[:, :, None]
    
    # Pz2
    diff2 = zf_b - z2_grid[:, :, None]
    valid_mask = z2_grid >= 0
    Pz2_mat = np.zeros_like(diff2)
    Pz2_mat[valid_mask, :] = gaussian_val(diff2[valid_mask, :], sigmaz_b)
    norm2 = np.trapz(Pz2_mat, x=chi, axis=-1)
    norm2[norm2 == 0] = 1.0 
    Pz2_mat /= norm2[:, :, None]

    # C. Construct Full Kernel Az (Nz, Npi, 400)
    with np.errstate(divide='ignore', invalid='ignore'):
        inv_chi2 = 1.0 / (chi**2)
        inv_chi2[np.isinf(inv_chi2)] = 0.0
    Az_mat = inv_chi2[None, None, :] * Pz1_mat * Pz2_mat
    
    # *** CRITICAL OPTIMIZATION ***
    # Integrate Kernel over Pi axis HERE
    # Reduces shape from (Nz, Npi, 400) -> (Nz, 400)
    Pi_max = block['LOS_bin', 'Pi_max'] if pi_mask_max < 0.0 else pi_mask_max
    pi_mask = (Pi >= -Pi_max) & (Pi <= Pi_max)
    
    Az_projected = np.trapz(Az_mat[:, pi_mask, :], x=Pi[pi_mask], axis=1)

    # D. Limber Integration via Matrix Multiplication
    dchi = np.diff(chi)
    weights = np.zeros_like(chi)
    weights[1:-1] = 0.5 * (dchi[:-1] + dchi[1:])
    weights[0] = 0.5 * dchi[0]
    weights[-1] = 0.5 * dchi[-1]
    
    P_weighted = P_gg_2d * weights[None, :]
    
    # Result: Projected C_ell for each z_low bin (Nz, Nell)
    Cell_projected = np.dot(Az_projected, P_weighted.T)
            
    if timing:
        T2 = time()
        print('Limber & Pi-Integration done. Starting Hankel...')

    # --- 5. Optimized Hankel Transform ---
    # Loop Nz times (200) instead of 20,000
    
    rp = np.logspace(np.log10(0.1), np.log10(300), 300)
    xi_projected = np.zeros((Nz, len(rp)))
    x0_arr = chi_of_z_spline(z_low)
    
    for i in range(Nz):
        theta_radians = np.arctan(rp / x0_arr[i])
        Cell_z = Cell_projected[i, :]

        # FFTLog for mu=0 (Galaxy Clustering)
        theta_new, xi_new = FFTLog(ell, Cell_z * ell, 0, 0, lowring=True)
        
        # Normalization
        xi_new = xi_new / theta_new / 2 / np.pi
        
        # Interpolate
        xi_int = np.interp(theta_radians, theta_new, xi_new, left=xi_new[0], right=xi_new[-1])
        
        xi_projected[i, :] = xi_int

    # --- 6. Final Redshift Integration ---
    
    # Get n(z) weights
    dz = zf[1] - zf[0]
    dxdz = np.gradient(chi, dz)
    nz_b_full = get_nz_on_grid(block, sample_b, zf)
    nz_a_full = get_nz_on_grid(block, sample_a, zf)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        X_kernel = nz_a_full * nz_b_full / (chi**2) / dxdz
        X_kernel[np.isinf(X_kernel)] = 0.0
        X_kernel[np.isnan(X_kernel)] = 0.0
    X_kernel[0] = 0.0
    
    trapz_norm = np.trapz(X_kernel, x=zf)
    W = np.zeros_like(X_kernel) if trapz_norm == 0 else X_kernel / trapz_norm
    
    # Interpolate W to z_low
    W_at_zlow = np.interp(z_low, zf, W)
    
    # Explicit K matrix to prevent dimension collapse
    Nell_rp = xi_projected.shape[1]
    K = np.tile(W_at_zlow[:, None], (1, Nell_rp))
    
    w_rp = np.trapz(xi_projected * K, x=z_low, axis=0)
    
    # Save
    block['galaxy_w','w_rp_1_1_%s_%s'%(sample_a,sample_b)] = w_rp
    block['galaxy_w', 'r_p'] = rp
    
    if timing:
        T3 = time()
        print('Total Time:', T3-T0)
    
    return 0

# --- Helper Functions ---
def gaussian_val(diff, s):
    return np.exp( -diff**2 / (2*s**2) )
            
def get_nz_on_grid(block, sample_name, z_grid):
    bin_idx = 1 
    name = 'nz_%s' % sample_name
    if not block.has_section(name):
         return np.zeros_like(z_grid)
    z_orig = block[name, 'z']
    nz_orig = block[name, 'bin_%d' % bin_idx]
    return np.interp(z_grid, z_orig, nz_orig, left=0, right=0)