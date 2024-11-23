cd $HOME

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh ###下载miniconda
chmod +x Miniconda3-latest-Linux-x86_64.sh  
./Miniconda3-latest-Linux-x86_64.sh   ###安装

source ~/.bashrc
conda init

conda create -n mp python=3.10 -y
conda activate mp
