from vllm import LLM, SamplingParams

prompts = [
    "Hello, how are you?",
    "What is the capital of France?",
    "What is the capital of Egypt?",
    "What is the capital of Japan?",
]


sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
llm = LLM(
    model="meta-llama/Llama-3.1-8B",
    tensor_parallel_size=2,
    # disable_custom_all_reduce=True,  # Add this line to disable custom all-reduce
    # gpu_memory_utilization=0.5,      # Add this to better manage memory
    quantization="fp8",
    enforce_eager=True,
    max_model_len=2048

)

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

