from vllm import LLM, SamplingParams
import time

# Sample prompts.
prompts = [
    # "Self-attention is an essential component of large language models (LLM) but a significant source of inference latency for long sequences. ",
    # "In multi-tenant LLM serving scenarios, the compute and memory operation cost of self-attention can be optimized by using the probability that multiple LLM requests have shared system prompts in prefixes.",
    "The capital of France is",
    # "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0, top_p=1)

# Create an LLM.
# llm = LLM(model="/root/models/opt-125m", gpu_memory_utilization=0.4)
llm = LLM(model="/root/models/opt-125m", 
          use_v2_block_manager=True,
          enable_prefix_caching=True,
          mp_enable=True, mp_host="10.210.22.244", mp_port="32765")
print("================================")
print("Begin to generate first")
print("================================")
prompts = [
    "In multi-tenant LLM serving scenarios, the compute and memory operation cost of self-attention can be optimized by using the probability that multiple LLM requests have shared system prompts in prefixes.",
]
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

print("================================")
print("Begin to generate second")
print("================================")
prompts = [
    "Self-attention is an essential component of large language models (LLM) but a significant source of inference latency for long sequences. ",
]
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

print("================================")
print("Begin to generate third")
print("================================")
prompts = [
    "The capital of France is",
]
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")