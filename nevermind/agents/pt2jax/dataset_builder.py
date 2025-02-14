import os
import ast
import json
import logging
import datasets

try:
    from github import Github, Auth
except ModuleNotFoundError:
    raise ValueError(
        "\x1b[31mDataset builder requires `PyGithub` package. " +
        "Install it using `pip install PyGithub`\x1b[0m"
    )


def get_default_logger(name: str = "nevermind", level=logging.DEBUG):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    console = logging.StreamHandler()
    console.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    if len(logger.handlers) == 0:
        # don't add handler if it's already added. useful for experiments in REPL
        logger.addHandler(console)
    return logger


logger = get_default_logger()

##-----------------------------------------------------------------------------------------


def repo_iterator(g: Github, repo_name: str):
    repo = g.get_repo(repo_name)
    contents = repo.get_contents("")
    while contents:
        fc = contents.pop(0)
        if fc.type == "file":
            yield fc
        elif fc.type == "dir":
            contents.extend(repo.get_contents(fc.path))



def extract_snippets(source: str):
    def extract_code(defn):
        start_line = defn.lineno
        end_line = defn.end_lineno
        lines = source.splitlines()[start_line - 1:end_line]
        return "\n".join(lines)
    tree = ast.parse(source)
    # functions and classes often depend on imported modules, so we collect imports first
    imports = [extract_code(node) for node in tree.body
               if isinstance(node, ast.Import | ast.ImportFrom)]
    all_imports = "\n".join(imports)
    # now we collect definitions of top-level functions and classes
    defs = [extract_code(node) for node in tree.body
            if isinstance(node, ast.FunctionDef | ast.ClassDef)]
    # finally, we enrich definitions with imports
    return [f"{all_imports}\n\n{defn}" for defn in defs]


def collect_snippets_with_meta(repo_name: str):
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        raise ValueError(f"GITHUB_TOKEN environment variable must be set")
    g = Github(auth=Auth.Token(token))
    for fc in repo_iterator(g, repo_name):
        if fc.path.endswith(".py"):
            # sources.append()
            source = fc.decoded_content.decode("utf-8")
            if "torch" not in source:
                # probably, this file doesn't contain PyTorch code
                continue
            for snippet in extract_snippets(source):
                yield {"snippet": snippet, "repo": repo_name, "path": fc.path}


PT_REPOSITORIES = [
    "pytorch/examples",
    "y0ast/pytorch-snippets"
]

def build_pytorch_dataset(dataset_name: str = "dfdx/pytorch-snippets"):
    snippets_with_meta = []
    total = 0
    for repo in PT_REPOSITORIES:
        logger.info(f"Working on repository {repo}")
        for i, swm in enumerate(collect_snippets_with_meta(repo)):
            snippets_with_meta.append(swm)
            total += 1
            if total % 10 == 0:
                logger.info(f"Collected {total} snippets ({i + 1} in this repo)")
    dataset = datasets.Dataset.from_list(snippets_with_meta)
    dataset.push_to_hub(dataset_name)
    logger.info(f"Finished. {total} code snippets pushed to HF hub at {dataset_name}")


