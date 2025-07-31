from vllm import LLM

llm = LLM(model="/workspace/BAGEL-7B-MoT", trust_remote_code=True)