### Direct IA forecast pipeline and covariance matrix computation.
This work builds on an optimized version of CosmoSIS pipeline developed by Simon Samuroff. He also provided the echoIA data format script. 
Contact: Zepei Yang (yang.zep@northeastern.edu)

#### How to use
Before using this repository, please ensure you have installed [direct_ia_theory](https://github.com/ssamuroff/direct_ia_theory). Then add 
command "export IA_LIB1 = /XXX/XXX/" in direct_ia_theory setup.sh, here XXX is your repo location. Or modified setup.sh in this project.

In this example, the configuration file generate-data.ini uses the CosmoSIS test sampler to generate a 3×2pt FITS data vector that follows the [echoIA data format](https://github.com/echo-IA/IAmeasurementsStore/tree/main/code).
The intrinsic alignment model can be chosen between NLA and TATT.
Model parameters are specified in values-generate.ini, primarily within the modules: intrinsic_alignment_parameters & bias_forecast_sample_density.
This setup is intended for generating mock samples with spectroscopic redshift information. The configuration file generate-data-photoz.ini also uses the CosmoSIS 
test sampler to generate a 3×2pt FITS file following the echoIA data format. This setup is intended for photometric samples. Either the NLA or TATT intrinsic 
alignment model can be used. The photo-z error model is configured in values-generate.ini through the photoz_errors module.

#### Scripts
add_nz:
* **Summary**: Replace the redshift distribution in the reference FITS file with that of the target sample.
* **Language**: Python
* **Inputs**: density sample and shape sample redshift distribution.
* **Outputs**: N/A

covmat:
* **Summary**: Compute the covariance matrix for the data vector, with units of $Mpc^2/h^2$. Input parameters should in Mpc.
* **Language**: Python
* **Inputs**: density sample and shape sample redshift distribution, nonlinear matter power spectrum, survey configuration, IA and galaxy bias parameters,
separate distance.
* **Outputs**: covariance matrix correspond to input separate distance. Order in Wgp-Wpp-Wgg.

makefits:
* **Summary**: Combine the output from the CosmoSIS test sampler into a FITS file, following the format of the echoIA data format script.
* **Language**: Python
* **Inputs**: Output fits name, $P_{g+}(k,z_{\text{eff}})$, $P_{=+}(k,z_{\text{eff}})$, $P_{gg}(k,z_{\text{eff}})$, shape noise, mean comoving number density.
* **Outputs**: 2pt fits file

photoz:
* **Summary**: Apply a simple Gaussian photo-$z$ error model to the matter power spectrum.
* **Language**: Python
* **Inputs**: photoz error $\sigma_z$. The photo-z module assumes a simple Gaussian form for the photometric redshift error.
* **Outputs**: $r_p$, $w_{i,j},\ \text{i,j} \in [g,+]$

nonlinear_bias/projection/structure/tatt:
* **Summary**: Functions the same as the structure/fast_pt in direct_ia_theory, but caches the intermediate data to improve execution speed.
* **Language**: Python

synthetic_parameter:
* **Summary**: The COSMOS2025 catalog fulfills the requirements for the Roman Space Telescope survey, consequently, we 
utilize it as the basis for our sample selection. We apply standard quality cuts on signal-to-noise, resolution, and 
shape noise to mimic realistic observational constraints. This allows us to construct robust mock catalogs, including 
spectroscopic samples, as well as photometric redMaGiC-like samples. For our parameter modeling, we use the power law 
equation of rest frame luminance to model the Intrinsic Alignment amplitude. We calculate this A_IA value individually for the red galaxy 
sub-samples based on their rest-frame absolute magnitudes. The linear galaxy bias is determined by linearly extrapolating the emission 
line galaxy and redMaGiC-like samples presented in [Dark Energy Survey Year 3 results: Galaxy-halo connection from galaxy-galaxy lensing]
(https://arxiv.org/pdf/2106.08438) and [Linear bias forecasts for emission line cosmological surveys]
(https://academic.oup.com/mnras/article/486/4/5737/5486120?guestAccessKey=). The non-linear galaxy bias is derived based 
on the fitting relation from simulations: $b_{2}=0.412-2.134b_{1}+0.929b_{1}^{2}+0.008b_{1}^{3}$.
* **Language**: ipynb



















