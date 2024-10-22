import os

from vllm import LLM, SamplingParams

# enable torch profiler, can also be set on cmd line
os.environ["VLLM_TORCH_PROFILER_DIR"] = "/mnt/user_storage/yue/profile"

model_name = "mistralai/Pixtral-12B-2409"
sampling_params = SamplingParams(max_tokens=8192)

# Lower max_num_seqs or max_model_len on low-VRAM GPUs.
llm = LLM(model=model_name, tokenizer_mode="mistral", max_model_len=8192)

llm.start_profile()

prompt = "Describe this image in one sentence."
image_url = "https://picsum.photos/id/237/200/300"

messages = [
    {
        "role":
        "user",
        "content": [
            {
                "type": "text",
                "text": prompt
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": image_url
                }
            },
        ],
    },
]
outputs = llm.chat(messages, sampling_params=sampling_params)

llm.stop_profile()

# Print the outputs.
print(outputs[0].outputs[0].text)
