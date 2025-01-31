# This script is run by buildkite to run the benchmarks and upload the results to buildkite

set -ex
set -o pipefail

# cd into parent directory of this file
cd "$(dirname "${BASH_SOURCE[0]}")/.."

(which wget && which curl) || (apt-get update && apt-get install -y wget curl)

# run python-based benchmarks and upload the result to buildkite
# python3 benchmarks/benchmark_latency.py --output-json latency_results.json 2>&1 | tee benchmark_latency.txt
# bench_latency_exit_code=$?

# python3 benchmarks/benchmark_throughput.py --input-len 256 --output-len 256 --output-json throughput_results.json 2>&1 | tee benchmark_throughput.txt
# bench_throughput_exit_code=$?

# MODEL_PATH="facebook/opt-125m"
MODEL_PATH="/root/model/meta-llama/Llama-3.1-8B-Instruct"
DATA_PATH=$HOME/ShareGPT_V3_unfiltered_cleaned_split.json

# run server-based benchmarks and upload the result to buildkite
# python3 -m vllm.entrypoints.openai.api_server \
#     --model ${MODEL_PATH} \
#     --enable-chunked-prefill False \
#     --gpu-memory-utilization 0.8 &
python3 -m vllm.entrypoints.openai.api_server \
    --model ${MODEL_PATH} \
    --gpu-memory-utilization 0.8 \
    --enable-prefix-caching \
    --enable-chunked-prefill False \
    --tensor-parallel-size 1 \
    --pipeline-parallel-size 1 \
    --enforce-eager \
    --max-model-len 128 \
    --max-num-seqs 8 \
    --use-v2-block-manager \
    --mp-enable --mp_host "172.16.253.12" --mp_port "3389" &
    # --mp-enable --mp_host "localhost" --mp_port "9999" &
server_pid=$!
# wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json

# wait for server to start, timeout after 600 seconds
timeout 600 bash -c 'until curl localhost:8000/v1/models > /dev/null 2>&1; do sleep 1; done' || exit 1
python3 benchmarks/benchmark_serving.py \
    --backend vllm \
    --dataset-name sharegpt \
    --dataset-path ${DATA_PATH} \
    --model ${MODEL_PATH} \
    --num-prompts 10 \
    --endpoint /v1/completions \
    --tokenizer ${MODEL_PATH} \
    2>&1 | tee benchmark_serving.txt
bench_serving_exit_code=$?
kill $server_pid

# write the results into a markdown file
# echo "### Latency Benchmarks" >> benchmark_results.md
# sed -n '1p' benchmark_latency.txt >> benchmark_results.md # first line
# echo "" >> benchmark_results.md
# sed -n '$p' benchmark_latency.txt >> benchmark_results.md # last line

# echo "### Throughput Benchmarks" >> benchmark_results.md
# sed -n '1p' benchmark_throughput.txt >> benchmark_results.md # first line
# echo "" >> benchmark_results.md
# sed -n '$p' benchmark_throughput.txt >> benchmark_results.md # last line

echo "### Serving Benchmarks" >> benchmark_results.md
sed -n '1p' benchmark_serving.txt >> benchmark_results.md # first line
echo "" >> benchmark_results.md
echo '```' >> benchmark_results.md
tail -n 24 benchmark_serving.txt >> benchmark_results.md # last 24 lines
echo '```' >> benchmark_results.md

# if the agent binary is not found, skip uploading the results, exit 0
if [ ! -f /usr/bin/buildkite-agent ]; then
    exit 0
fi

# upload the results to buildkite
buildkite-agent annotate --style "info" --context "benchmark-results" < benchmark_results.md

# exit with the exit code of the benchmarks
if [ $bench_latency_exit_code -ne 0 ]; then
    exit $bench_latency_exit_code
fi

if [ $bench_throughput_exit_code -ne 0 ]; then
    exit $bench_throughput_exit_code
fi

if [ $bench_serving_exit_code -ne 0 ]; then
    exit $bench_serving_exit_code
fi

rm ShareGPT_V3_unfiltered_cleaned_split.json
buildkite-agent artifact upload "*.json"
