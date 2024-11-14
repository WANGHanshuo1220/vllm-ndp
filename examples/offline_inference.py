from vllm import LLM, SamplingParams
import time

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0, top_p=1)

# Create an LLM.
llm = LLM(model="/root/models/opt-125m", gpu_memory_utilization=0.4)
# llm = LLM(model="/root/models/opt-125m",
#           mp_enable=True, mp_host="10.210.22.246", mp_port="32403")
# llm = LLM(model="/root/models/opt-125m",
#           mp_enable=True, mp_host="localhost", mp_port="9999",
#           gpu_memory_utilization=0.4)
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
print("================================")
print("Begin to generate")
print("================================")
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
