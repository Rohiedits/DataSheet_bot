# HT67F4892 Datasheet Expert — Streamlit

A Streamlit RAG application for the internally bundled HT67F4892 Rev. V1.20
datasheet.

## What it does

- The 182-page datasheet is bundled internally under `data/`.
- Users do NOT upload the PDF for every question.
- Page-aware chunks are embedded and searched with FAISS.
- Retrieved passages are supplied to the LLM.
- The LLM is instructed to answer only from the datasheet context.
- Answers include PDF page references.
- Users can open a preview of the cited original PDF page.
- Conversation history supports follow-up questions.
- Low temperature is used to reduce unsupported generation.

## Architecture

PDF -> PyMuPDF -> page-aware chunks -> Sentence Transformers -> FAISS
-> retrieve relevant passages -> LLM -> explanation + references -> Streamlit

## Setup on Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Create `.env` from `.env.example` and add your API key:

```text
OPENAI_API_KEY=...
OPENAI_MODEL=gpt-4o-mini
```

Build the vector index once:

```powershell
python build_index.py
```

Start the app:

```powershell
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal.

## Example questions

- Explain Timer 0 in simple terms.
- What is the ADC resolution?
- What does TM0C0 bit 5 do?
- What are the operating voltage ranges?
- Explain the UART receiver.
- Give the relevant register configuration for a timer question.
- Where in the datasheet is the ADC conversion timing described?

## Grounding behavior

The prompt explicitly tells the LLM not to fill missing information with general
knowledge. If retrieval is insufficient, the assistant should say it cannot
reliably answer from the provided datasheet.

## Important production note

The default app uses an LLM API. The PDF remains bundled with the app, but
retrieved text is sent to the configured LLM provider. For a fully local/private
deployment, replace the `ask_llm()` function with a locally hosted model.

## Next improvements

For the strongest datasheet assistant, add:
- hybrid keyword + vector retrieval
- reranking
- register/table-aware chunking
- OCR/image extraction for scanned diagrams
- exact source-section labels
- authentication for admin/index management
- feedback/evaluation set for retrieval accuracy
