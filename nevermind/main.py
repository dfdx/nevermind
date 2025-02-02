import jax.numpy as jnp
from fabrique.models.llm import LLM
from smolagents import CodeAgent, DuckDuckGoSearchTool, HfApiModel



PT_SNIPPET = '''
def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")
'''


def main():
    llm = LLM.from_pretrained(
        "meta-llama/CodeLlama-7b-hf",
        max_batch_sise=1,
        max_seq_len=1024,
        dtype=jnp.bfloat16,
        param_dtype=jnp.bfloat16
    )
    out = llm.generate("Write a Python function to print current time in HH:MM:ss format")
    print(out)

    agent = CodeAgent(tools=[DuckDuckGoSearchTool()], model=HfApiModel())
    agent.run(f"Rewrite the following PyTorch snippet into equivalent code in Flax NNX (not Flax Linen!). Snippet: {PT_SNIPPET}")

