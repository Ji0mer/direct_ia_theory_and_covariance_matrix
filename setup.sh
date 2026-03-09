# source cosmosis, replace these lines with your cosmosis
cd /projects/blazek_group_storage/zepei/
source /projects/blazek_group_storage/zepei/miniconda/bin/activate
conda activate ./env
source cosmosis-configure
cd -

# you need this line unless you want camb to be very slow
export OMP_NUM_THREADS=1

# replace these lines with wherever the repos live in your setup
export COSMOSIS_LIB=/projects/blazek_group_storage/zepei/cosmosis-standard-library/
export IA_LIB=/projects/blazek_group_storage/zepei/ia_forecast/direct_ia_theory/
export IA_LIB1=/projects/blazek_group_storage/zepei/ia_forecast/direct_ia_to_public/
export DATA_DIR=$IA_LIB1/fits_data/
