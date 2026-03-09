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
* **Inputs**:
* **Outputs**:

covmat:
* **Summary**: Compute the covariance matrix for the data vector, with units of $Mpc^2/h^2$.
* **Language**: Python
* **Inputs**:
* **Outputs**:

makefits:
* **Summary**: Combine the output from the CosmoSIS test sampler into a FITS file, following the format of the echoIA data format script.
* **Language**: Python
* **Inputs**:
* **Outputs**:

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



















