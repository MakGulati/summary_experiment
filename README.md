# Summary Experiment: Repository Analysis and Toy Example Generator

This project uses LlamaIndex and various LLMs to analyze GitHub repositories and generate toy example Jupyter notebooks demonstrating key concepts from the analyzed repository.

## Features

- Clone GitHub repositories
- Analyze repository contents using LlamaIndex
- Generate toy examples using configurable LLMs
- Create Jupyter notebooks from generated examples

## Requirements

- Python 3.9+
- See `requirements.txt` for required packages

## Setup

1. Clone this repository:
   ```
   git clone <this-repo-url>
   cd summary_experiment
   ```

2. Install required packages:
   ```
   pip install -r requirements.txt
   ```

3. Create a `keys.env` file in the project root and add your LLM API key(s):
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ANTHROPIC_API_KEY=your_anthropic_api_key_here
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

## Usage

Run the script with a GitHub repository URL and LLM choice as arguments:

