from vllm import LLM

llm = LLM(model="microsoft/swin-tiny-patch4-window7-224",
          trust_remote_code=True)
