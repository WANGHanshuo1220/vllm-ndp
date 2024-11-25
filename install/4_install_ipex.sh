echo "4_install_ipex.sh"
cd $HOME

eval "$($HOME/miniconda3/bin/conda shell.bash hook)" > /dev/null 2>&1
conda init bash
conda activate mp

mkdir ipex
cd ipex
cp $HOME/vllm-mp/install/compile_bundle.sh .
bash compile_bundle.sh