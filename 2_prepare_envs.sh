cd $HOME/vllm-ndp
conda activate mp

sudo apt-get update  -y
sudo apt-get install -y gcc-12 g++-12 libnuma-dev
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 100 --slave /usr/bin/g++ g++ /usr/bin/g++-12

pip install cmake==3.26 wheel packaging ninja "setuptools-scm>=8" numpy
pip install -v -r requirements-cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu

export PATH=$PATH:~/.local/bin
export PATH=/usr/bin:$PATH
source ~/.bashrc

pip install setuptools==61