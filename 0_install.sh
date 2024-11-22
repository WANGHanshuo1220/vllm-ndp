bash 1_prepare_basic.sh
bash 2_prepare_envs.sh
bash 3_install_dnnl.sh &
bash 4_install_ipex.sh &

wait

cd $HOME/vllm-ndp
conda activate mp
VLLM_TARGET_DEVICE=cpu python setup.py install