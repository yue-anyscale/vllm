import os

from vllm import LLM, SamplingParams
from vllm.assets.image import ImageAsset

# enable torch profiler, can also be set on cmd line
os.environ["VLLM_TORCH_PROFILER_DIR"] = "/mnt/user_storage/yue/tmp"

model_name = "Qwen/Qwen2-VL-7B-Instruct"

llm = LLM(model=model_name, max_model_len=8192, max_num_seqs=5)
llm.start_profile()

stop_token_ids = None
question = "What is the content of this image?"
prompt = ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
            f"{question}<|im_end|>\n"
            "<|im_start|>assistant\n")
image = ImageAsset("cherry_blossom") \
            .pil_image.convert("RGB")
inputs = {
    "prompt": prompt,
    "multi_modal_data": {
        "image": image
    },
}
sampling_params = SamplingParams(temperature=0.2, max_tokens=64, stop_token_ids=stop_token_ids)
outputs = llm.generate(inputs, sampling_params=sampling_params)

llm.stop_profile()

# Print the outputs.
for o in outputs:
    generated_text = o.outputs[0].text
    print(generated_text)
