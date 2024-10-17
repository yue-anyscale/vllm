# Basic workflow learnings

## Use cases

**Offline Inference workload**
- one example usage is in offline_inference_XXX.py:
```
llm = LLM(model="meta-llama/Meta-Llama-3-8B-Instruct")
```
which is starting the server, and then run the workload using this started server. How does this work under the hood?
is it like this:
```
// step 1 - when we run python offline_inference_xxx.py it runs LLM()
// As we call LLM(), the server is started as a separate process, running in the background. This allows it to handle requests independently of the main process.
llm = LLM(model="meta-llama/Meta-Llama-3-8B-Instruct")

// step 2 - when it runs llm.chat() which is sending request
// This will send request to the started server
llm.chat(messages) // this will process the messages to prompts, and call llm.generate()
// llm.generate(prompts)

// step 3 - when the script finished run, the LLM class has a mechanism to clean up resources, including shutting down the background server
```


**Online Inference workload using OpenAI API**

run vllm serve in a console:
example `vllm serve meta-llama/Meta-Llama-3-8B-Instruct`

run the python script in another console, it will sends request to the server.
// example - a script running openai api
client = OpenAI(local server address, keyword)
client.chat.completion.create(messages) // vllm provides http openAI completion and chat api, it will send the messages as payload to the vllm server, the backend will call the vllm.chat under the hood.

The python code is using OpenAI API, so that customers don't need to change their code.


## Detailed Code Walkthrough

### Request life journey - llm.generate()
Learning from examples/offline_inference.py.
When we call llm.generate(), what is happening under the hood?

User-side Example:
```
// Build sampling params.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
// Create LLM (specify model, etc)
llm = LLM(model="facebook/opt-125m")
// Call generate
outputs = llm.generate(prompts, sampling_params)
```

What happens when we call llm.generate(...):
```
// step 1 - validate and add request to the inference engine.
generate() // in entrypoint/
->
add_request() // in engine/
->
preprocess requests // in engine/
-> 
add_seg_group // in scheduler/

// step 2 - run engine
run_engine() // in entrypoint/
->
step() // in engine/
-> 
execute_model() // in executor/ 
->
execute_model() // in worker/
-> 
return outputs

// step 3 - validate outputs
validate_outputs() // in engine/
```

[P1][Scheduler] questions:
- term - messages, prompts, sequence, seq_group. What's their differences?
```
from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained("neuralmagic/Meta-Llama-3-8B-Instruct-FP8")
messages = [
	{"role": "system", "content": "You are an assistant"},
	{"role": "user", "content": "What's your name?"}
]
t.apply_chat_template(messages)
t.apply_chat_template(messages, tokenize=False)
```
encourage use messages as input.
prompt: has bos & eos, etc, a processed string for the message
sequence: see the sequence struct, it includes not only prompt, tokens, status (prefill, decode), etc
seq_group: will deprecate inside engine, will be handled in entrypoint /client side, it's for the sampler (n, best_of)

- What is the preprocess prompt -> seg_group do? what extra info added to the prompt? solved above
- What is the scheduler do when we add seg_group (a prompt) to the waiting queue? How's the scheduler work?
    - every coming request will trigger schedule(), which process the waiting requests
    - default scheduler: prioritize prefill, will not mix prefill and decode
    - chunked prefill scheduler: mix prefill and decode, not waste bandwidth
    - budget: 16386, max batch size, e.g. 2000 tokens per prompt, 16386 can process 16386 / 2000 prompts/
- there is a loop in run_engine() which waits for all unfinished_requests output, why this doesn't include the requests not coming from the current call?
    - The run_engine() in entrypoint is actually offline, only the offline use case call this.
    - the online use case call the http server, which will not call that
    - The online and offline meets in engine.
- The request seems goes through a long journey, from entrypoint -> engine -> scheduler -> run, what's the latency of the scheduler?
    - Yes, previously 
        - the API server cost a lot (as it's in the same process as the executor), now removed
        - the scheduler cost, the multi step (Will implemented) helped reduce the schedule times, now solved
        - preprocess, prompt, seq_group.
- what data type (precision) it's using by default? what quantization it supports?
    - it's depends on model, see the hugging face page - model config - dtype - most bf16
- quantization
    - vllm quantization doc
    ```
    [1.8, 2.5] simple example of quantize to int:
[18, 25], scale=0.1

[2.345678, 3.4556778] more complex example:
[0, 3.4556778-2.345678]*scale

[[...], [...], [...]] // also have this cases, store more scales and offset

// For some case like w8a16: weights in int8, activation in fp16
// as activation is more hard to quantize (because they are not static, they are runtime determined) - sometimes use calibration set to determine if we want to quantize activation.
// for matmul w8a16, a just read as fp16, weights use the scale and offset like:
w16 = (w8 * scale) + offset
// to convert to w16, and then use the tensor core fp16 to calculate
a16
// the result is usually float, we convert to fp16 as the output for next step activation usage.
o32 -> o16


w8a16
w8a8
    ```


[P1][Executor] questions:
- Code point & how do we determine which executor to use when we call execute_model()?
- Which model_runner we are calling in worker/ for each model? e.g. llama3 8B, pixtual, etc

### llm.chat() and batch inference
Learning from examples/offline_inference_chat.py.
Example of not batch inference:
```
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
```

Example of batch inference: 
```
conversations = [conversation for _ in range(10)]

outputs = llm.chat(messages=conversations,
                   sampling_params=sampling_params,
                   use_tqdm=True)
```

- llm.generate() - when we give a list of prompts, it will automatically batch inference for the prompts
- llm.chat() accepted message types - list of dict (conversation), list of list of dict (conversations)
  llm.chat() will calls llm.generate() under the hood, if provided conversations, it will preprocess them to prompts and pass into llm/generate().



