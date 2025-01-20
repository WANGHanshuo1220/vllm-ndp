python vllm/mem_pool/rdma/server_rdma.py \
    --max-model-len 128 \
    --max-num-seqs 8 \
    --swap-space 1 \
    --tensor-parallel 4 \
    --pipeline-parallel 2 \
    --model /root/model/meta-llama/Llama-3.1-8B-Instruct/ 