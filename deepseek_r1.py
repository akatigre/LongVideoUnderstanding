from vllm import LLM, SamplingParams

def load_model():
    model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    llm = LLM(model_id, tensor_parallel_size=4)
    return llm, None

def generate(llm, prompt):
    params = SamplingParams(temperature=0.7, max_tokens=32768)
    outputs = llm.generate(prompt, params)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
    return outputs