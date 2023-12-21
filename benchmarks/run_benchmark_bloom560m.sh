cd /software/users/mdvoretckii/huda
source reset.sh
cd /software/users/mdvoretckii/habana_vllm
python -m pip install -e .
python -m pip install xformers --no-deps
cd benchmarks
#python benchmark_throughput.py --tokenizer bigscience/bloom-560m --model bigscience/bloom-560m --dataset ShareGPT_V3_unfiltered_cleaned_split.json --num-prompts 100
python benchmark_throughput.py --tokenizer lmsys/vicuna-7b-v1.3 --model lmsys/vicuna-7b-v1.3 --dataset ShareGPT_V3_unfiltered_cleaned_split.json --num-prompts 100
#curl -X POST  -H "Accept: Application/json" -H "Content-Type: application/json" http://localhost:8000/generate -d '{"prompt":"Would you like a jelly baby?","use_beam_search":false,"n":1}'


# Missing ops:
# Bloom: alibi
# llama: RMS Norm, RoPE, fused silu, fail in sample
# ---
# GPT2: gelu_new
# Aquila: issues with external source
# Baichuan: no tokenizer
# Falcon: fail in sample
# Falcon RW: TypeError: memory_efficient_attention_forward() missing 1 required positional argument: 'cu_seq_lens'
# GPT BigCode: gated, santacoder fails in sample (not affected by CPU RoPE)
# GPT-J: gelu_new
# GPT-NeoX: gelu_fast
# InternLM: no tokenizer class
# Mistral: max_num_batched_tokens (2048) is smaller than max_model_len (32768).
# MPT: TypeError: memory_efficient_attention_forward() missing 1 required positional argument: 'cu_seq_lens'
# OPT: fail in sample
# Qwen: no tokenizer class
