cd $HOME
conda activate mp
mkdir ipex && cd ipex
wget https://github.com/intel/intel-extension-for-pytorch/raw/v2.5.0%2Bcpu/scripts/compile_bundle.sh
bash compile_bundle.sh