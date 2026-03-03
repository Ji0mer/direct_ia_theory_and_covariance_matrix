# source cosmosis, replace these lines with your cosmosis
cd /projects/blazek_group_storage/zepei/
source /projects/blazek_group_storage/zepei/miniconda/bin/activate
conda activate ./env
source cosmosis-configure
cd -

# you need this line unless you want camb to be very slow
export OMP_NUM_THREADS=1

# replace these lines with wherever the repos live in your setup
export COSMOSIS_LIB=
export IA_LIB=
export IA_LIB1=
export DATA_DIR=$IA_LIB1/fits_data/
