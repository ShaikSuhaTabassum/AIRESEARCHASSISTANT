# AI Research Assistant

This project generates structured research reports using Gemini and LangChain.

## Features

- Gather research papers
- Extract formulas
- Identify trends
- Structured output using Pydantic

## Installation

```bash
pip install -r requirements.txt
```

Create a `.env` file:

```text
GOOGLE_API_KEY=your_api_key
```

Run:

```bash
python main.py
py -m pip install streamlit
py -m pip install arxiv
py -m pip install reportlab
py -m streamlit run app.py
```