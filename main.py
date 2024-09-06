"""
Summary Experiment: Repository Analysis and Toy Example Generator

This script clones a given GitHub repository, analyzes its contents using
LlamaIndex and a specified LLM, and generates a toy example Jupyter notebook
demonstrating the key concepts of the repository.

Usage:
    python main.py <git_repo_url> --llm [openai|anthropic|gemini]

Requirements:
    See requirements.txt for a list of required packages.
"""

import os
import sys
import shutil
import warnings
import argparse
from typing import Optional

import torch
import faiss
import nbformat
from dotenv import load_dotenv
from git import Repo
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Document
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.storage.storage_context import StorageContext
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.llms.anthropic import Anthropic
from llama_index.llms.gemini import Gemini
from nbformat.v4 import new_markdown_cell, new_code_cell, new_notebook

# Load environment variables
load_dotenv("keys.env")

# Suppress specific warning
# Suppress specific warning
warnings.filterwarnings(
    "ignore", message="^`clean_up_tokenization_spaces` was not set.*"
)

def get_llm(llm_choice: str):
    """
    Get the specified LLM instance.

    Args:
        llm_choice (str): The chosen LLM ('openai', 'anthropic', or 'gemini').

    Returns:
        LLM instance
    """
    if llm_choice == 'openai':
        return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    elif llm_choice == 'anthropic':
        return Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    elif llm_choice == 'gemini':
        return Gemini(api_key=os.getenv("GEMINI_API_KEY"))
    else:
        raise ValueError(f"Unsupported LLM choice: {llm_choice}")

def clone_repo(repo_url: str, local_path: str) -> None:
    """
    Clone a GitHub repository to a local path.

    Args:
        repo_url (str): URL of the GitHub repository to clone.
        local_path (str): Local path where the repository will be cloned.
    """
    Repo.clone_from(repo_url, local_path)

def get_repo_contents(repo_path: str) -> Optional[VectorStoreIndex]:
    """
    Process the contents of a cloned repository and create a VectorStoreIndex.

    Args:
        repo_path (str): Path to the cloned repository.

    Returns:
        Optional[VectorStoreIndex]: The created index, or None if an error occurred.
    """
    print(f"Starting to process repository at {repo_path}")

    try:
        reader = SimpleDirectoryReader(input_dir=repo_path)
        documents = reader.load_data()
        print(f"Loaded {len(documents)} files from the repository")

        parser = SimpleNodeParser.from_defaults()
        nodes = parser.get_nodes_from_documents(documents)
        print(f"Parsed {len(nodes)} nodes from the documents")

        d = 384  # dimension for the default 'sentence-transformers/all-MiniLM-L6-v2' model
        faiss_index = faiss.IndexFlatL2(d)
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        index = VectorStoreIndex(
            nodes, storage_context=storage_context, embed_model=embed_model
        )
        print("Created VectorStoreIndex successfully")

        return index

    except Exception as e:
        print(f"Error processing repository contents: {e}")
        return None

def generate_toy_example(query_engine) -> str:
    """
    Generate a toy example using the specified LLM.

    Args:
        query_engine: The query engine to use for generating the toy example.

    Returns:
        str: The generated toy example as a string.
    """
    toy_example_query = query_engine.query(
        """Analyze the given code repository and create a toy example Jupyter notebook that demonstrates its key concepts. Include the following:

            1. A brief introduction explaining what the notebook demonstrates
            2. Import necessary libraries
            3. Create a simple dataset or input (if applicable)
            4. Implement core functionality from the project
            5. Demonstrate how to use the main features
            6. Add comments explaining each step
            7. If applicable, visualize results or output

            Present your response as a series of markdown and code cells, clearly indicating which is which. Use '# [markdown]' for markdown cells and '# [code]' for code cells."""
    )
    return toy_example_query.response

def create_jupyter_notebook(toy_example: str) -> None:
    """
    Create a Jupyter notebook from the generated toy example.

    Args:
        toy_example (str): The generated toy example string.
    """
    nb = new_notebook()
    cells = toy_example.split("\n# [")

    for cell in cells:
        if cell.startswith("markdown]"):
            content = cell[9:].strip()
            nb.cells.append(new_markdown_cell(content))
        elif cell.startswith("code]"):
            content = cell[5:].strip()
            nb.cells.append(new_code_cell(content))

    with open("toy_example.ipynb", "w") as f:
        nbformat.write(nb, f)

    print("\nToy example notebook saved as 'toy_example.ipynb'")

def main(repo_url: str, llm_choice: str) -> None:
    """
    Main function to process a GitHub repository and generate a toy example.

    Args:
        repo_url (str): URL of the GitHub repository to analyze.
        llm_choice (str): The chosen LLM to use.
    """
    temp_dir = "temp_repo"

    try:
        clone_repo(repo_url, temp_dir)
        repo_index = get_repo_contents(temp_dir)

        if repo_index:
            llm = get_llm(llm_choice)
            query_engine = repo_index.as_query_engine(llm=llm)

            toy_example = generate_toy_example(query_engine)
            print("\nToy Example:")
            print(toy_example)

            create_jupyter_notebook(toy_example)

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze a GitHub repository and generate a toy example.")
    parser.add_argument("repo_url", help="URL of the GitHub repository to analyze")
    parser.add_argument(
        "--llm",
        choices=["openai", "anthropic", "gemini"],
        default="gemini",
        help="LLM to use for analysis",
    )
    args = parser.parse_args()

    main(args.repo_url, args.llm)
