# version w/o smolagents

import re
import importlib
import traceback
from dataclasses import dataclass
from typing import Dict, List

import jax.numpy as jnp
from flax import nnx
from fabrique import LLM, ChatMessage



SYSTEM_PROMPT = """
You are a multistep AI agent that can translate PyTorch code to the corresponding JAX/Flax NNX code.

To solve the task, you must plan forward to proceed in a series of steps. At each step, you will be
provided with:

1. PyTorch code between <code:pytorch> and </code:pytorch> tags.
2. Information from the previous steps between <memory> and </memory> tags.

You must write the next step, which can be one of

  * tool use in format <tool>tool_name(tool_args)</tool>
  * resulting JAX/Flax NNX code between <code:jax> and </code:jax> tags


You have access to the following tools:

* help(qualified.function.name) - returns the docstring of the qualified.function.name.
    Use this tool when you are unsure about function's signature.

* search(query) - find APIs and code snippets related to the topic described in the query.
    Use this tool when you don't know for sure how to express something in JAX/Flax NNX.


Here are a few examples:

---

<code:pytorch>
import torch

x = torch.rand((2, 3))
<code:pytorch>

<memory>
</memory>

<tool>
help(jax.random.uniform)
</tool>

---

<code:pytorch>
import torch

x = torch.rand((2, 3))
<code:pytorch>

<memory>
uniform(key: 'ArrayLike', shape: 'Shape' = (), dtype: 'DTypeLikeFloat' = <class 'float'>, minval: 'RealArray' = 0.0, maxval: 'RealArray' = 1.0) -> 'Array'
    Sample uniform random values in [minval, maxval) with given shape/dtype.

    Args:
      key: a PRNG key used as the random key.
      shape: optional, a tuple of nonnegative integers representing the result
        shape. Default ().
</memory>

<code:jax>
import jax

key = jax.random.key(0)
x = jax.random.uniform(key, (2, 3))
</code:jax>

---

Now begin!


"""



# <think>
# This PyTorch code generates a random array of size (2, 3). In JAX, the equivalent function
# is jax.random.uniform(), but I'm not sure about it's format. Thus, I will use tool `help`
# to learn more about this function's signature
# </think>


# <think>
# This PyTorch code generates a random array of size (2, 3). In JAX, the equivalent function
# is jax.random.uniform(), and contents of memory tells that jax.random.uniform() takes
# two arguments - key, which is a PRNG key, and shape, which in our case is (2, 3). To
# create the PRNG key I can use function jax.random.key().
# </think>


INPUT = """
<code:pytorch>
import torch.nn as nn

x = torch.rand(10).reshape(2, 5)
</code:pytorch>
"""

INPUT2 = """
<code:pytorch>
```python
import torch.nn as nn

conv = nn.Conv2d(3, 32, 5, stride=1)
x = torch.rand((1, 3, 16, 16))
y = conv(x)
```
</code:pytorch>

<memory>
</memory>

"""



INPUT3 = """
<code:pytorch>
```python
import torch.nn as nn

conv = nn.Conv2d(3, 32, 5, stride=1)
x = torch.rand((1, 3, 16, 16))
y = conv(x)
```
</code:pytorch>

<memory>
Help on function `jax.nn.Conv2D` returned error:

```
AttributeError                            Traceback (most recent call last)
Cell In[9], line 1
----> 1 help(jax.nn.Conv2D)

AttributeError: module 'jax.nn' has no attribute 'Conv2D'
```
</memory>
"""


def help_tool(fn_name):
    try:
        mod_name, name = fn_name.rsplit(".", 1)
        mod = importlib.import_module(mod_name)
        fn = getattr(mod, name)
        doc = fn.__doc__
        return f"""Help on function {fn_name}:\n\n{doc}"""
    except Exception as e:
        tb = traceback.format_exc()
        return f"""Help on function {fn_name} resulted in an error:\n\n {tb}"""



class FormatException(Exception):
    pass


@dataclass
class TraceEntry:
    prompt: str
    response: str


class Agent:

    def __init__(self, llm: LLM, tools=(help_tool,)):
        self.llm = llm
        self.tools = tools
        self.memory: List[str] = []
        self.trace: List[TraceEntry] = []

    def reset(self):
        self.memory = []
        self.trace = []


    def step(self, user_prompt: str):
        memory = "<memory>\n" + "\n\n".join(self.memory) + "\n</memory>"
        prompt = f"{user_prompt}\n\n{memory}\n\n"
        messages = [
            ChatMessage(role="system", content=SYSTEM_PROMPT),
            ChatMessage(role="user", content=prompt)
        ]
        print("------------------------------------------")
        print(self.llm.apply_chat_template(messages))
        print("\n\n")
        response = self.llm.generate(messages).content
        self.trace.append((prompt, response))
        if m := re.search(r"<tool>(.*)</tool>", response, re.DOTALL):
            tool_str = m.groups()[0].strip()
            tm = re.match(r"^([a-zA-Z0-9_]+)\((.*)\)$", tool_str)
            if not tm:
                raise FormatException(f"Couldn't parse tool call from string: {tool_str}")
            tool_name, tool_args = tm.groups()
            assert tool_name == "help"  # TODO replace with lookup in tool registry
            tool_output = help_tool(tool_args)
            self.memory.append(tool_output)
            return None
        elif m := re.search(r"<code:jax>(.*)</code:jax>", response, re.DOTALL):
            return m.groups()[0]
        else:
            raise FormatException(f"Cannot parse response:\n\n {response}")


def main():
    llm = LLM.from_pretrained(
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        max_batch_size=1,
        max_seq_len=4096,
        dtype=jnp.bfloat16,
        param_dtype=jnp.bfloat16
    )

    self = Agent(llm)
    prompt = """<code:pytorch>
    import torch.nn as nn

    conv = nn.Conv2d(3, 32, 5, stride=1)
    x = torch.rand((1, 3, 16, 16))
    y = conv(x)
    </code:pytorch>"""



# TODO:
# * sft model on JAX/Flax NNX code




# TASK = """
#     Rewrite the following PYTORCH CODE to FLAX NNX CODE. Return the FLAX NNX CODE.
#     Example:


#     PYTORCH CODE:

#     ```python
#     import torch.nn as nn
#     import torch.nn.functional as F

#     class Net(nn.Module):
#         def __init__(self):
#             super(Net, self).__init__()
#             self.conv = nn.Conv2d(1, 32, 3, 1)

#         def forward(self, x):
#             x = self.conv(x)
#             x = F.relu(x)
#             output = F.log_softmax(x, dim=1)
#             return output
#     ```

#     FLAX NNX CODE:

#     ```python
#     import flax.nnx as nnx

#     class Net(nnx.Module):
#         def __init__(self):
#             super(Net, self).__init__()
#             self.conv = nnx.Conv(1, 32, (3, 3), stride=1)

#         def __call__(self, x):
#             x = self.conv(x)
#             x = nnx.relu(x)
#             output = nnx.log_softmax(x, dim=1)
#             return output
#     ```


#     ---

#     Now, apply this logic to the following code.

#     PYTORCH CODE:

#     ```
#     import torch

#     x = torch.rand(3, 4)
#     y = torch.rand(4, 3)
#     z = x @ y
#     ```
#     """


