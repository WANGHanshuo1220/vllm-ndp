echo "3_install_dnnl.sh"
cd $HOME

eval "$($HOME/miniconda3/bin/conda shell.bash hook)" > /dev/null 2>&1
conda init bash
conda activate mp

git clone git@github.com:WANGHanshuo1220/oneDNN.git
cd $HOME/oneDNN
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX ..
make -j$(nproc)
make install