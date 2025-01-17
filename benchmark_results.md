### Serving Benchmarks
Namespace(backend='vllm', base_url=None, host='localhost', port=8000, endpoint='/v1/completions', dataset=None, dataset_name='sharegpt', dataset_path='/root/ShareGPT_V3_unfiltered_cleaned_split.json', model='/root/model/meta-llama/Llama-3.1-8B-Instruct', tokenizer='/root/model/meta-llama/Llama-3.1-8B-Instruct', best_of=1, use_beam_search=False, num_prompts=1, sharegpt_output_len=None, sonnet_input_len=550, sonnet_output_len=150, logprobs=None, sonnet_prefix_len=200, random_input_len=1024, random_output_len=128, random_range_ratio=1.0, random_prefix_len=0, request_rate=inf, seed=0, trust_remote_code=False, disable_tqdm=False, profile=False, save_result=False, metadata=None, result_dir=None, result_filename=None, percentile_metrics='ttft,tpot,itl', metric_percentiles='99')

```
Initial test run completed. Starting main benchmark run...
Traffic request rate: inf
  0%|          | 0/1 [00:00<?, ?it/s]100%|██████████| 1/1 [00:03<00:00,  3.67s/it]100%|██████████| 1/1 [00:03<00:00,  3.67s/it]
============ Serving Benchmark Result ============
Successful requests:                     1         
Benchmark duration (s):                  3.67      
Total input tokens:                      4         
Total generated tokens:                  6         
Request throughput (req/s):              0.27      
Output token throughput (tok/s):         1.63      
Total Token throughput (tok/s):          2.72      
---------------Time to First Token----------------
Mean TTFT (ms):                          665.56    
Median TTFT (ms):                        665.56    
P99 TTFT (ms):                           665.56    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          601.37    
Median TPOT (ms):                        601.37    
P99 TPOT (ms):                           601.37    
---------------Inter-token Latency----------------
Mean ITL (ms):                           601.35    
Median ITL (ms):                         637.94    
P99 ITL (ms):                            841.05    
==================================================
```
### Serving Benchmarks
Namespace(backend='vllm', base_url=None, host='localhost', port=8000, endpoint='/v1/completions', dataset=None, dataset_name='sharegpt', dataset_path='/root/ShareGPT_V3_unfiltered_cleaned_split.json', model='/root/model/meta-llama/Llama-3.1-8B-Instruct', tokenizer='/root/model/meta-llama/Llama-3.1-8B-Instruct', best_of=1, use_beam_search=False, num_prompts=1, sharegpt_output_len=None, sonnet_input_len=550, sonnet_output_len=150, logprobs=None, sonnet_prefix_len=200, random_input_len=1024, random_output_len=128, random_range_ratio=1.0, random_prefix_len=0, request_rate=inf, seed=0, trust_remote_code=False, disable_tqdm=False, profile=False, save_result=False, metadata=None, result_dir=None, result_filename=None, percentile_metrics='ttft,tpot,itl', metric_percentiles='99')

```
Initial test run completed. Starting main benchmark run...
Traffic request rate: inf
  0%|          | 0/1 [00:00<?, ?it/s]100%|██████████| 1/1 [00:09<00:00,  9.89s/it]100%|██████████| 1/1 [00:09<00:00,  9.89s/it]
============ Serving Benchmark Result ============
Successful requests:                     1         
Benchmark duration (s):                  9.89      
Total input tokens:                      4         
Total generated tokens:                  6         
Request throughput (req/s):              0.10      
Output token throughput (tok/s):         0.61      
Total Token throughput (tok/s):          1.01      
---------------Time to First Token----------------
Mean TTFT (ms):                          1353.23   
Median TTFT (ms):                        1353.23   
P99 TTFT (ms):                           1353.23   
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          1707.38   
Median TPOT (ms):                        1707.38   
P99 TPOT (ms):                           1707.38   
---------------Inter-token Latency----------------
Mean ITL (ms):                           1707.36   
Median ITL (ms):                         1637.22   
P99 ITL (ms):                            2024.69   
==================================================
```
### Serving Benchmarks
Namespace(backend='vllm', base_url=None, host='localhost', port=8000, endpoint='/v1/completions', dataset=None, dataset_name='sharegpt', dataset_path='/root/ShareGPT_V3_unfiltered_cleaned_split.json', model='/root/model/meta-llama/Llama-3.1-8B-Instruct', tokenizer='/root/model/meta-llama/Llama-3.1-8B-Instruct', best_of=1, use_beam_search=False, num_prompts=1, sharegpt_output_len=None, sonnet_input_len=550, sonnet_output_len=150, logprobs=None, sonnet_prefix_len=200, random_input_len=1024, random_output_len=128, random_range_ratio=1.0, random_prefix_len=0, request_rate=inf, seed=0, trust_remote_code=False, disable_tqdm=False, profile=False, save_result=False, metadata=None, result_dir=None, result_filename=None, percentile_metrics='ttft,tpot,itl', metric_percentiles='99')

```
Initial test run completed. Starting main benchmark run...
Traffic request rate: inf
  0%|          | 0/1 [00:00<?, ?it/s]100%|██████████| 1/1 [00:00<00:00,  4.44it/s]100%|██████████| 1/1 [00:00<00:00,  4.44it/s]
============ Serving Benchmark Result ============
Successful requests:                     1         
Benchmark duration (s):                  0.23      
Total input tokens:                      4         
Total generated tokens:                  6         
Request throughput (req/s):              4.44      
Output token throughput (tok/s):         26.61     
Total Token throughput (tok/s):          44.35     
---------------Time to First Token----------------
Mean TTFT (ms):                          40.80     
Median TTFT (ms):                        40.80     
P99 TTFT (ms):                           40.80     
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          36.84     
Median TPOT (ms):                        36.84     
P99 TPOT (ms):                           36.84     
---------------Inter-token Latency----------------
Mean ITL (ms):                           36.80     
Median ITL (ms):                         36.78     
P99 ITL (ms):                            37.18     
==================================================
```
### Serving Benchmarks
Namespace(backend='vllm', base_url=None, host='localhost', port=8000, endpoint='/v1/completions', dataset=None, dataset_name='sharegpt', dataset_path='/root/ShareGPT_V3_unfiltered_cleaned_split.json', model='/root/model/meta-llama/Llama-3.1-8B-Instruct', tokenizer='/root/model/meta-llama/Llama-3.1-8B-Instruct', best_of=1, use_beam_search=False, num_prompts=1, sharegpt_output_len=None, sonnet_input_len=550, sonnet_output_len=150, logprobs=None, sonnet_prefix_len=200, random_input_len=1024, random_output_len=128, random_range_ratio=1.0, random_prefix_len=0, request_rate=inf, seed=0, trust_remote_code=False, disable_tqdm=False, profile=False, save_result=False, metadata=None, result_dir=None, result_filename=None, percentile_metrics='ttft,tpot,itl', metric_percentiles='99')

```
Initial test run completed. Starting main benchmark run...
Traffic request rate: inf
  0%|          | 0/1 [00:00<?, ?it/s]100%|██████████| 1/1 [00:13<00:00, 13.73s/it]100%|██████████| 1/1 [00:13<00:00, 13.73s/it]
============ Serving Benchmark Result ============
Successful requests:                     1         
Benchmark duration (s):                  13.73     
Total input tokens:                      4         
Total generated tokens:                  6         
Request throughput (req/s):              0.07      
Output token throughput (tok/s):         0.44      
Total Token throughput (tok/s):          0.73      
---------------Time to First Token----------------
Mean TTFT (ms):                          1902.13   
Median TTFT (ms):                        1902.13   
P99 TTFT (ms):                           1902.13   
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          2365.45   
Median TPOT (ms):                        2365.45   
P99 TPOT (ms):                           2365.45   
---------------Inter-token Latency----------------
Mean ITL (ms):                           2365.43   
Median ITL (ms):                         2392.87   
P99 ITL (ms):                            2436.97   
==================================================
```
### Serving Benchmarks
Namespace(backend='vllm', base_url=None, host='localhost', port=8000, endpoint='/v1/completions', dataset=None, dataset_name='sharegpt', dataset_path='/root/ShareGPT_V3_unfiltered_cleaned_split.json', model='/root/model/meta-llama/Llama-3.1-8B-Instruct', tokenizer='/root/model/meta-llama/Llama-3.1-8B-Instruct', best_of=1, use_beam_search=False, num_prompts=1, sharegpt_output_len=None, sonnet_input_len=550, sonnet_output_len=150, logprobs=None, sonnet_prefix_len=200, random_input_len=1024, random_output_len=128, random_range_ratio=1.0, random_prefix_len=0, request_rate=inf, seed=0, trust_remote_code=False, disable_tqdm=False, profile=False, save_result=False, metadata=None, result_dir=None, result_filename=None, percentile_metrics='ttft,tpot,itl', metric_percentiles='99')

```
Initial test run completed. Starting main benchmark run...
Traffic request rate: inf
  0%|          | 0/1 [00:00<?, ?it/s]100%|██████████| 1/1 [00:09<00:00,  9.33s/it]100%|██████████| 1/1 [00:09<00:00,  9.33s/it]
============ Serving Benchmark Result ============
Successful requests:                     1         
Benchmark duration (s):                  9.33      
Total input tokens:                      4         
Total generated tokens:                  6         
Request throughput (req/s):              0.11      
Output token throughput (tok/s):         0.64      
Total Token throughput (tok/s):          1.07      
---------------Time to First Token----------------
Mean TTFT (ms):                          1965.86   
Median TTFT (ms):                        1965.86   
P99 TTFT (ms):                           1965.86   
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          1473.51   
Median TPOT (ms):                        1473.51   
P99 TPOT (ms):                           1473.51   
---------------Inter-token Latency----------------
Mean ITL (ms):                           1473.48   
Median ITL (ms):                         1469.33   
P99 ITL (ms):                            1713.68   
==================================================
```
### Serving Benchmarks
Namespace(backend='vllm', base_url=None, host='localhost', port=8000, endpoint='/v1/completions', dataset=None, dataset_name='sharegpt', dataset_path='/root/ShareGPT_V3_unfiltered_cleaned_split.json', model='/root/model/meta-llama/Llama-3.1-8B-Instruct', tokenizer='/root/model/meta-llama/Llama-3.1-8B-Instruct', best_of=1, use_beam_search=False, num_prompts=1, sharegpt_output_len=None, sonnet_input_len=550, sonnet_output_len=150, logprobs=None, sonnet_prefix_len=200, random_input_len=1024, random_output_len=128, random_range_ratio=1.0, random_prefix_len=0, request_rate=inf, seed=0, trust_remote_code=False, disable_tqdm=False, profile=False, save_result=False, metadata=None, result_dir=None, result_filename=None, percentile_metrics='ttft,tpot,itl', metric_percentiles='99')

```
Initial test run completed. Starting main benchmark run...
Traffic request rate: inf
  0%|          | 0/1 [00:00<?, ?it/s]100%|██████████| 1/1 [00:11<00:00, 11.56s/it]100%|██████████| 1/1 [00:11<00:00, 11.56s/it]
============ Serving Benchmark Result ============
Successful requests:                     1         
Benchmark duration (s):                  11.56     
Total input tokens:                      4         
Total generated tokens:                  7         
Request throughput (req/s):              0.09      
Output token throughput (tok/s):         0.61      
Total Token throughput (tok/s):          0.95      
---------------Time to First Token----------------
Mean TTFT (ms):                          2346.32   
Median TTFT (ms):                        2346.32   
P99 TTFT (ms):                           2346.32   
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          1535.09   
Median TPOT (ms):                        1535.09   
P99 TPOT (ms):                           1535.09   
---------------Inter-token Latency----------------
Mean ITL (ms):                           1842.08   
Median ITL (ms):                         1870.74   
P99 ITL (ms):                            2314.00   
==================================================
```
