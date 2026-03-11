# AI Prompt Template

This project is a small prompt-engineering assignment centered on a chainable Python prompt template system. It includes a reusable `PromptTemplate` class, an extended `FeynmanTemplate` for the Feynman learning technique, example scripts, and Jupyter notebooks for experimentation.

## Project Overview

- `given/prompt_template.py` defines the core prompt-building classes.
- `given/prompt_template_examples.py` shows how to build prompts and optionally send them to Google Gemini.
- `given/PromptTemplate.ipynb` and `given/Feynman_Technique_Example.ipynb` provide notebook-based examples.
- `technique_research.md` explains the reasoning behind choosing the Feynman learning technique.

The main idea is that prompts can be built incrementally with method chaining. For example, you can start with a base prompt and then add a role, context, examples, constraints, or Feynman-style refinement steps before generating the final prompt text.

## Requirements

- Python 3
- `google-genai`
- `python-dotenv`
- A Google Gemini API key stored in a `.env` file as `GOOGLE_API_KEY=your_key_here`

## Setup

From the project root, install the required packages:

```powershell
pip install google-genai python-dotenv
```

If you want to run the example that calls Gemini, make sure your `.env` file contains:

```env
GOOGLE_API_KEY=your_key_here
```

## Running the Code

Run the example script from the `given/` folder:

```powershell
cd given
python prompt_template_examples.py
```

By default, the script is set to:

```python
generate_prompt_only = True
```

That means it will print the constructed prompts without making an API request.

If you want the script to generate model output as well, open `given/prompt_template_examples.py` and change:

```python
generate_prompt_only = False
```

Then run the script again:

```powershell
cd given
python prompt_template_examples.py
```

## Running the Notebooks

To explore the notebooks interactively:

```powershell
jupyter notebook
```

Then open either of these files:

- `given/PromptTemplate.ipynb`
- `given/Feynman_Technique_Example.ipynb`

## Notes

- The example script imports `prompt_template` from the same folder, so it should be run while your shell is inside `given/`.
- The default model in the example script is `gemini-2.5-flash`.
- If `generate_prompt_only` is set to `False`, the script will require a valid API key and internet access.
