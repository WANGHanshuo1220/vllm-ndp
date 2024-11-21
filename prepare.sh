ssh-keygen -t rsa -b 4096 -C "hanshuo_wang@163.com"

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh ###下载miniconda
chmod +x Miniconda3-latest-Linux-x86_64.sh  
./Miniconda3-latest-Linux-x86_64.sh   ###安装

git clone git@github.com:WANGHanshuo1220/vllm-ndp.git
git clone git@github.com:oneapi-src/oneDNN.git

conda create -n mp python=3.10 -y
conda activate mp

cd oneDNN
mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX ..
make -j$(nproc)
make install