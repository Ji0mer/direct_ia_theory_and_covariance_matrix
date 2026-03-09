### Direct IA forecast pipeline and covariance matrix computation.
This work builds on an optimized version of CosmoSIS pipeline developed by Simon Samuroff. He also provided the echoIA data format script. 
Contact: Zepei Yang (yang.zep@northeastern.edu)

#### How to use
Before using this repository, please ensure you have installed [direct_ia_theory](https://github.com/ssamuroff/direct_ia_theory). Then add 
command "export IA_LIB1 = /XXX/XXX/" in direct_ia_theory setup.sh, here XXX is your repo location. Or modified setup.sh in this project.

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

nonlinear_bias:
* **Summary**: Functions the same as the nonlinear bias in direct_ia_theory, but caches the intermediate data to improve execution speed.
* **Language**: Python
* **Inputs**:
* **Outputs**:

photoz:
* **Summary**: Apply a simple Gaussian photo-$z$ error model to the matter power spectrum.
* **Language**: Python
* **Inputs**:
* **Outputs**:

projection:
* **Summary**: Functions the same as the projection/projected_corrs_legendre in direct_ia_theory, but caches the intermediate data to improve execution speed.
* **Language**: Python
* **Inputs**:
* **Outputs**:

structure:
* **Summary**: Functions the same as the structure/fast_pt in direct_ia_theory, but caches the intermediate data to improve execution speed.
* **Language**: Python
* **Inputs**:
* **Outputs**:

tatt:
* **Summary**: ...
* **Language**: Python
* **Inputs**:
* **Outputs**:



















