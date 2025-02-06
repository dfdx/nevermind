import jax
import jax.numpy as jnp
from flax import nnx
from fabrique.models.llm import LLM
from smolagents import CodeAgent, DuckDuckGoSearchTool, HfApiModel, ChatMessage



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


LLAMA_TMPL = """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful coding agent<|eot_id|><|start_header_id|>user<|end_header_id|>

{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""


class ModelCaller:
    def __init__(self, llm=None):
        self.llm = llm or LLM.from_pretrained(
            # "meta-llama/CodeLlama-7b-hf",
            "meta-llama/Meta-Llama-3-8B-Instruct",
            max_batch_size=1,
            max_seq_len=4096,
            dtype=jnp.bfloat16,
            param_dtype=jnp.bfloat16
        )
        self.rngs = nnx.Rngs(0)

    def __call__(self, inst: str, **kwargs):
        global X
        X = inst
        print(f"INSTRUCTION:\n\n{inst}\n\n")
        print(f"ModelCaller: extra kwargs = {kwargs}")
        prompt = LLAMA_TMPL.format(inst)
        generated = self.llm.generate(
            prompt,
            max_length=self.llm.model.args.max_seq_len,
            temperature=1.0,
            # top_p=0.5,
            # top_k=3,
            prng_key=self.rngs()
        )
        msg = ChatMessage(role="assistant", content=generated)
        return msg


def main():
    # jax.config.update("jax_explain_cache_misses", True)
    mcall = ModelCaller()
    prompt = "Write a Python function to print current time in HH:MM:SS format"
    out = mcall(prompt)
    print(out.content)

    #= HfApiModel() ==#
    agent = CodeAgent(tools=[DuckDuckGoSearchTool()], model=mcall, additional_authorized_imports="*")
    # out = agent.run(f"Rewrite the following PyTorch snippet into equivalent code in Flax NNX (not Flax Linen!). Output function definition as a string.\n\nSnippet: {PT_SNIPPET}")
    out = agent.run(f"How many times is elephant more heavy than a mouse?")
    print(out)
