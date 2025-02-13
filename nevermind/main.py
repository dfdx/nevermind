from typing import Dict, List

import jax
import jax.numpy as jnp
from flax import nnx
from fabrique import LLM, ChatMessage
from smolagents import CodeAgent, DuckDuckGoSearchTool, HfApiModel, ChatMessage as AChatMessage



SYSTEM_PROMPT = """You are an expert assistant who can solve any task using code blobs. You will be given a task to solve as best you can.
To do so, you have been given access to a list of tools: these tools are basically Python functions which you can call with code.
To solve the task, you must plan forward to proceed in a series of steps, in a cycle of 'Thought:', 'Code:', and 'Observation:' sequences.

At each step, in the 'Thought:' sequence, you should first explain your reasoning towards solving the task and the tools that you want to use.
Then in the 'Code:' sequence, you should write the code in simple Python. The code sequence must end with '<end_code>' sequence.
During each intermediate step, you can use 'print()' to save whatever important information you will then need.
These print outputs will then appear in the 'Observation:' field, which will be available as input for the next step.
In the end you have to return a final answer using the `final_answer` tool.

Here are a few examples using notional tools:
---
{examples}

Above example were using notional tools that might not exist for you. On top of performing computations in the Python code snippets that you create, you only have access to these tools:

{{tool_descriptions}}

{{managed_agents_descriptions}}

Here are the rules you should always follow to solve your task:
1. Always provide a 'Thought:' sequence, and a 'Code:\n```py' sequence ending with '```<end_code>' sequence, else you will fail.
2. Use only variables that you have defined!
3. Always use the right arguments for the tools. DO NOT pass the arguments as a dict as in 'answer = wiki({'query': "What is the place where James Bond lives?"})', but use the arguments directly as in 'answer = wiki(query="What is the place where James Bond lives?")'.
4. Take care to not chain too many sequential tool calls in the same code block, especially when the output format is unpredictable. For instance, a call to search has an unpredictable return format, so do not have another tool call that depends on its output in the same block: rather output results with print() to use them in the next block.
5. Call a tool only when needed, and never re-do a tool call that you previously did with the exact same parameters.
6. Don't name any new variable with the same name as a tool: for instance don't name a variable 'final_answer'.
7. Never create any notional variables in our code, as having these in your logs might derail you from the true variables.
8. You can use imports in your code, but only from the following list of modules: {{authorized_imports}}
9. The state persists between code executions: so if in one step you've created variables or imported modules, these will all persist.
10. Don't give up! You're in charge of solving the task, not providing directions to solve it.

Now Begin! If you solve the task correctly, you will receive a reward of $1,000,000.
"""


MY_SYSTEM_PROMPT = """You are an expert assistant who can solve any task using Python code snippets.
Your output should consist of 2 sections:

1. "Thought:", where you describe what you are going to do.
2. "Code:", where you write Python code to solve the task.

The code section should start with "```python\n" and end with "```".
The last statement in code should always be "print(asnwer)", where variable "answer"
should store the result of computation.

A few examples:

---

User input: What is 2 + 2?
Your output:

Thought: This looks like a mathematical expression. To calculate it, I can use built-in Python arithmetics.
Code:
```python
answer = 2 + 2
print(answer)
```

---

User input: What is the title of the main Wikipedia page today?
Your output:

Thought: First, I need to download the main page of Wikipedia using built-in Python packages.
Then print whatever is inside of <title> and </title> tags.
Code:
```python
# import needed Python modules
import requests
import re

# download main page of Wikipedia
response = requests.get("https://wikipedia.org")

# extract title
text = response.text
answer = re.match(r"<title>(.*)</title>", text).groups(0)
```

"""


def agent_dicts_to_chat_messages(adicts: List[Dict]):
    def adict_to_chat_message(adict: Dict):
        content = ""
        for c in adict["content"]:
            assert c["type"] == "text"
            content += c["text"] + "\n\n"
        return ChatMessage(
            role=adict["role"].value,
            content=content
        )
    return [adict_to_chat_message(adict) for adict in adicts]


I = None

class ModelCaller:
    def __init__(self, llm=None):
        self.llm = llm or LLM.from_pretrained(
            "meta-llama/Meta-Llama-3.1-8B-Instruct",
            max_batch_size=1,
            max_seq_len=4096,
            dtype=jnp.bfloat16,
            param_dtype=jnp.bfloat16
        )
        self.rngs = nnx.Rngs(0)

    def __call__(self, agent_dicts: List[Dict], **kwargs):
        global I
        if I is None:
            I = agent_dicts, kwargs
        messages = agent_dicts_to_chat_messages(agent_dicts)
        response = self.llm.generate(
            messages,
            max_length=self.llm.model.args.max_seq_len,
        )
        print(">" * 80 + f"\n\n{self.llm.apply_chat_template(messages)}\n\n")
        print("-" * 80)
        print(f"\n\n{self.llm.apply_chat_template([response])}\n\n" + "<" * 80 + "\n\n")
        msg = AChatMessage(role="assistant", content=response.content)
        return msg


def main():
    llm = LLM.from_pretrained(
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        max_batch_size=1,
        max_seq_len=4096,
        dtype=jnp.bfloat16,
        param_dtype=jnp.bfloat16
    )
    # mcall = ModelCaller(llm)

    # agent = CodeAgent(
    #     tools=[DuckDuckGoSearchTool()],
    #     model=mcall,
    #     additional_authorized_imports="*",
    #     system_prompt=SYSTEM_PROMPT
    # )
    # # out = agent.run(f"Rewrite the following PyTorch snippet into equivalent code in Flax NNX (not Flax Linen!). Output function definition as a string.\n\nSnippet: {PT_SNIPPET}")
    # out = agent.run(f"How many times is elephant more heavy than a mouse?")
    # print(out)

    # with open("inst.txt") as fp:
    #     inst = fp.read()
    # llm = mcall.llm
    # out = llm.generate(inst + "<|start_header_id|>assistant<|end_header_id|>\n\n")
    # print(out)

    # mcall = ModelCaller(llm)
    # print(mcall.llm.generate(INST, temperature=1.0, prng_key=mcall.rngs()))

    task = """
    Rewrite the following expression from PyTorch to JAX. Return the rewrittern code
    as a STRING in the variable `answer`:

    ```
    import torch

    x = torch.random(3, 4)
    y = torch.random(4, 3)
    z = x @ y
    ```
    """

    messages = [
        ChatMessage(role="system", content=MY_SYSTEM_PROMPT),
        ChatMessage(role="user", content=task)
    ]
    out = llm.generate(messages)
    print(out.content)

