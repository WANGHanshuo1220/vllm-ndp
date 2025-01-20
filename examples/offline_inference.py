from vllm import LLM, SamplingParams
import time
import argparse

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--o", type=int, default=0)
    args = parser.parse_args()

    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=0, top_p=1)

    # Create an LLM.
    llm = LLM(model="/root/model/meta-llama/Llama-3.1-8B-Instruct", 
    # llm = LLM(model="facebook/opt-125m",
              gpu_memory_utilization=0.8,
              use_v2_block_manager=True,
              enable_prefix_caching=True,
              enable_chunked_prefill=False,
              tensor_parallel_size=2,
              pipeline_parallel_size=1,
              max_model_len=64,
              max_num_seqs=8,
              worker_use_ray=False,
              distributed_executor_backend="mp",
              engine_id=0)
            #   mp_enable=True, mp_host="172.16.253.12", mp_port="3389")

    print("\n" + "#"*18 + f" {args.o}th order prompt " + "#"*18)
    if args.o == 0:
        prompts = [
            " Async output processing is only supported for CUDA or TPU. Disabling it for other platforms. CUDA graph is not supported on CPU, fallback to the eager mode.",
            "In multi-tenant LLM serving scenarios, the compute and memory operation cost of self-attention can be optimized by using the probability that multiple LLM requests have shared system prompts in prefixes.",
        ]
        for i in range(len(prompts)):
            print("" + "*"*18 + f" {i}th prompt " + "*"*18)
            outputs = llm.generate(prompts[i], sampling_params)
    else:
        time.sleep(10)
        prompts = [
            "The capital of France is",
            "In multi-tenant LLM serving scenarios, the compute and memory operation cost of self-attention can be optimized by using the probability that multiple LLM requests have shared system prompts in prefixes.",
        ]
        for i in range(len(prompts)):
            outputs = llm.generate(prompts[i], sampling_params)