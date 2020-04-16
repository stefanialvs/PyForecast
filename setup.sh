conda create --name pyforecast python=3.6
source ~/anaconda3/etc/profile.d/conda.sh
#source ~/miniconda/etc/profile.d/conda.sh
conda activate pyforecast

conda install -c anaconda numpy==1.16.1
conda install -c anaconda scipy==1.2.1
conda install -c anaconda scikit-learn==0.21.3
conda install -c anaconda pandas==0.25.2
conda install -c conda-forge matplotlib==3.1.1
conda install -c anaconda seaborn==0.9.0
conda install -c anaconda jupyter
conda install -c anaconda Pillow