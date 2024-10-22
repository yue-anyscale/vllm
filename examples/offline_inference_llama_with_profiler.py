import os

from vllm import LLM, SamplingParams

# enable torch profiler, can also be set on cmd line
os.environ["VLLM_TORCH_PROFILER_DIR"] = "/mnt/user_storage/yue/tmp"


llm = LLM(model="meta-llama/Meta-Llama-3-8B-Instruct")
llm.start_profile()
sampling_params = SamplingParams(temperature=0.5)


def print_outputs(outputs):
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    print("-" * 80)


print("=" * 80)

# In this script, we demonstrate how to pass input to the chat method:

conversation = [
    {
        "role": "system",
        "content": "You are a helpful assistant"
    },
    {
        "role": "user",
        "content": "Hello"
    },
    {
        "role": "assistant",
        "content": "Hello! How can I assist you today?"
    },
    {
        "role": "user",
        "content": "Write an essay about the importance of higher education.",
    },
]
outputs = llm.chat(conversation,
                   sampling_params=sampling_params,
                   use_tqdm=False)
llm.stop_profile()

print_outputs(outputs)

