# Prepare github setting
# ssh-keygen -t rsa -b 4096 -C "hanshuo_wang@163.com"
# cat ~/.ssh/id_rsa.pub
# echo "============================="

bash 1_prepare_basic.sh
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda init bash

bash 2_prepare_envs.sh
bash 3_install_dnnl.sh &
bash 4_install_ipex.sh &

wait

cd $HOME/vllm-ndp
conda activate mp
VLLM_TARGET_DEVICE=cpu python setup.py install