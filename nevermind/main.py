from typing import Dict, List

import jax
import jax.numpy as jnp
from flax import nnx
from fabrique import LLM, ChatMessage
from smolagents import CodeAgent, DuckDuckGoSearchTool, HfApiModel, ChatMessage as AChatMessage



INST = """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023
Today Date: 26 Jul 2024

You are an expert assistant who can solve any task using code blobs. You will be given a task to solve as best you can.
To do so, you have been given access to a list of tools: these tools are basically Python functions which you can call with code.
To solve the task, you must plan forward to proceed in a series of steps, in a cycle of 'Thought:', 'Code:', and 'Observation:' sequences.

At each step, in the 'Thought:' sequence, you should first explain your reasoning towards solving the task and the tools that you want to use.
Then in the 'Code:' sequence, you should write the code in simple Python. The code sequence must end with '<end_code>' sequence.
During each intermediate step, you can use 'print()' to save whatever important information you will then need.
These print outputs will then appear in the 'Observation:' field, which will be available as input for the next step.
In the end you have to return a final answer using the `final_answer` tool.

Here are a few examples using notional tools:
---
Task: "Generate an image of the oldest person in this document."

Thought: I will proceed step by step and use the following tools: `document_qa` to find the oldest person in the document, then `image_generator` to generate an image according to the answer.
Code:
```py
answer = document_qa(document=document, question="Who is the oldest person mentioned?")
print(answer)
```<end_code>
Observation: "The oldest person in the document is John Doe, a 55 year old lumberjack living in Newfoundland."

Thought: I will now generate an image showcasing the oldest person.
Code:
```py
image = image_generator("A portrait of John Doe, a 55-year-old man living in Canada.")
final_answer(image)
```<end_code>

---
Task: "What is the result of the following operation: 5 + 3 + 1294.678?"

Thought: I will use python code to compute the result of the operation and then return the final answer using the `final_answer` tool
Code:
```py
result = 5 + 3 + 1294.678
final_answer(result)
```<end_code>

---
Task:
"Answer the question in the variable `question` about the image stored in the variable `image`. The question is in French.
You have been provided with these additional arguments, that you can access using the keys as variables in your python code:
{'question': 'Quel est l'animal sur l'image?', 'image': 'path/to/image.jpg'}"

Thought: I will use the following tools: `translator` to translate the question into English and then `image_qa` to answer the question on the input image.
Code:
```py
translated_question = translator(question=question, src_lang="French", tgt_lang="English")
print(f"The translated question is {translated_question}.")
answer = image_qa(image=image, question=translated_question)
final_answer(f"The answer is {answer}")
```<end_code>

---
Task:
In a 1979 interview, Stanislaus Ulam discusses with Martin Sherwin about other great physicists of his time, including Oppenheimer.
What does he say was the consequence of Einstein learning too much math on his creativity, in one word?

Thought: I need to find and read the 1979 interview of Stanislaus Ulam with Martin Sherwin.
Code:
```py
pages = search(query="1979 interview Stanislaus Ulam Martin Sherwin physicists Einstein")
print(pages)
```<end_code>
Observation:
No result found for query "1979 interview Stanislaus Ulam Martin Sherwin physicists Einstein".

Thought: The query was maybe too restrictive and did not find any results. Let's try again with a broader query.
Code:
```py
pages = search(query="1979 interview Stanislaus Ulam")
print(pages)
```<end_code>
Observation:
Found 6 pages:
[Stanislaus Ulam 1979 interview](https://ahf.nuclearmuseum.org/voices/oral-histories/stanislaus-ulams-interview-1979/)

[Ulam discusses Manhattan Project](https://ahf.nuclearmuseum.org/manhattan-project/ulam-manhattan-project/)

(truncated)

Thought: I will read the first 2 pages to know more.
Code:
```py
for url in ["https://ahf.nuclearmuseum.org/voices/oral-histories/stanislaus-ulams-interview-1979/", "https://ahf.nuclearmuseum.org/manhattan-project/ulam-manhattan-project/"]:
    whole_page = visit_webpage(url)
    print(whole_page)
    print("
" + "="*80 + "
")  # Print separator between pages
```<end_code>
Observation:
Manhattan Project Locations:
Los Alamos, NM
Stanislaus Ulam was a Polish-American mathematician. He worked on the Manhattan Project at Los Alamos and later helped design the hydrogen bomb. In this interview, he discusses his work at
(truncated)

Thought: I now have the final answer: from the webpages visited, Stanislaus Ulam says of Einstein: "He learned too much mathematics and sort of diminished, it seems to me personally, it seems to me his purely physics creativity." Let's answer in one word.
Code:
```py
final_answer("diminished")
```<end_code>

---
Task: "Which city has the highest population: Guangzhou or Shanghai?"

Thought: I need to get the populations for both cities and compare them: I will use the tool `search` to get the population of both cities.
Code:
```py
for city in ["Guangzhou", "Shanghai"]:
    print(f"Population {city}:", search(f"{city} population")
```<end_code>
Observation:
Population Guangzhou: ['Guangzhou has a population of 15 million inhabitants as of 2021.']
Population Shanghai: '26 million (2019)'

Thought: Now I know that Shanghai has the highest population.
Code:
```py
final_answer("Shanghai")
```<end_code>

---
Task: "What is the current age of the pope, raised to the power 0.36?"

Thought: I will use the tool `wiki` to get the age of the pope, and confirm that with a web search.
Code:
```py
pope_age_wiki = wiki(query="current pope age")
print("Pope age as per wikipedia:", pope_age_wiki)
pope_age_search = web_search(query="current pope age")
print("Pope age as per google search:", pope_age_search)
```<end_code>
Observation:
Pope age: "The pope Francis is currently 88 years old."

Thought: I know that the pope is 88 years old. Let's compute the result using python code.
Code:
```py
pope_current_age = 88 ** 0.36
final_answer(pope_current_age)
```<end_code>

Above example were using notional tools that might not exist for you. On top of performing computations in the Python code snippets that you create, you only have access to these tools:


- web_search: Performs a duckduckgo web search based on your query (think a Google search) then returns the top search results.
    Takes inputs: {'query': {'type': 'string', 'description': 'The search query to perform.'}}
    Returns an output of type: string

- final_answer: Provides a final answer to the given problem.
    Takes inputs: {'answer': {'type': 'any', 'description': 'The final answer to the problem'}}
    Returns an output of type: any



Here are the rules you should always follow to solve your task:
1. Always provide a 'Thought:' sequence, and a 'Code:
```py' sequence ending with '```<end_code>' sequence, else you will fail.
2. Use only variables that you have defined!
3. Always use the right arguments for the tools. DO NOT pass the arguments as a dict as in 'answer = wiki({'query': "What is the place where James Bond lives?"})', but use the arguments directly as in 'answer = wiki(query="What is the place where James Bond lives?")'.
4. Take care to not chain too many sequential tool calls in the same code block, especially when the output format is unpredictable. For instance, a call to search has an unpredictable return format, so do not have another tool call that depends on its output in the same block: rather output results with print() to use them in the next block.
5. Call a tool only when needed, and never re-do a tool call that you previously did with the exact same parameters.
6. Don't name any new variable with the same name as a tool: for instance don't name a variable 'final_answer'.
7. Never create any notional variables in our code, as having these in your logs will derail you from the true variables.
8. You can use imports in your code, but only from the following list of modules: You can import from any package you want.
9. The state persists between code executions: so if in one step you've created variables or imported modules, these will all persist.
10. Don't give up! You're in charge of solving the task, not providing directions to solve it.

Now Begin! If you solve the task correctly, you will receive a reward of $1,000,000.<|eot_id|><|start_header_id|>user<|end_header_id|>

New task:
How many times is elephant more heavy than a mouse?<|eot_id|>
"""


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


# LLAMA_TMPL = """
# <|begin_of_text|><|start_header_id|>system<|end_header_id|>

# You are a helpful coding agent<|eot_id|><|start_header_id|>user<|end_header_id|>

# {}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
# """
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
            I = agent_dicts
        messages = agent_dicts_to_chat_messages(agent_dicts)
        response = self.llm.generate(
            messages,
            max_length=self.llm.model.args.max_seq_len,
        )
        print(response.role)
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
    out = llm.generate([ChatMessage(role="user", content="What time is it?")], max_length=128)
    print(out)

    out = llm.generate(INST)
    print(out)


    # jax.config.update("jax_explain_cache_misses", True)
    mcall = ModelCaller()
    mcall = ModelCaller(mcall.llm)  # reinit after code change
    prompt = "Write a Python function to print current time in HH:MM:SS format"
    out = mcall(prompt)
    print(out.content)

    #= HfApiModel() ==#
    agent = CodeAgent(tools=[DuckDuckGoSearchTool()], model=mcall, additional_authorized_imports="*")
    # out = agent.run(f"Rewrite the following PyTorch snippet into equivalent code in Flax NNX (not Flax Linen!). Output function definition as a string.\n\nSnippet: {PT_SNIPPET}")
    out = agent.run(f"How many times is elephant more heavy than a mouse?")
    print(out)

    with open("inst.txt") as fp:
        inst = fp.read()
    llm = mcall.llm
    out = llm.generate(inst + "<|start_header_id|>assistant<|end_header_id|>\n\n")
    print(out)

    mcall = ModelCaller(llm)
    print(mcall.llm.generate(INST, temperature=1.0, prng_key=mcall.rngs()))