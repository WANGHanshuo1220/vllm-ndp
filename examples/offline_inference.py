from vllm import LLM, SamplingParams
import time
import random

# Sample prompts.
prompts = [
    # "Self-attention is an essential component of large language models (LLM) but a significant source of inference latency for long sequences. ",
    # "In multi-tenant LLM serving scenarios, the compute and memory operation cost of self-attention can be optimized by using the probability that multiple LLM requests have shared system prompts in prefixes.",
    "The capital of France is",
    # "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0, top_p=1)

# mark
# Create an LLM.
llm = LLM(model="/root/model/meta-llama/Llama-3.1-8B-Instruct", 
# llm = LLM(model="facebook/opt-125m",
          use_v2_block_manager=True,
          enable_prefix_caching=True,
          enable_chunked_prefill=False,
          max_model_len=64,
          max_num_seqs=8,
          worker_use_ray=False,)
        #   mp_enable=True, mp_host="172.16.253.20", mp_port="3389")

prompts = [
    # "In multi-tenant LLM serving scenarios, the compute and memory operation cost of self-attention can be optimized by using the probability that multiple LLM requests have shared system prompts in prefixes.",
    "The capital of France is",
]
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

# n = 130 * 1000
# token_ids = [random.randint(0, 128000) for _ in range(n)]
# prompts = {
#     "prompt_token_ids": token_ids
# }
# outputs = llm.generate(prompts, sampling_params)
# # Print the outputs.
# for output in outputs:
#     prompt = output.prompt_token_ids
#     generated_text = output.outputs[0].text
#     print(f"Prompt: {len(prompt)}, Generated text: {generated_text!r}")