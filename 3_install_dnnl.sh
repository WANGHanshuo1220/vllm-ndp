cd $HOME
conda activate mp
git clone git@github.com:oneapi-src/oneDNN.git
cd oneDNN
mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX ..
make -j$(nproc)
make install