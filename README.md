# ragbench

End-to-end RAG evaluation for embedding models. The embedding model is the only thing that changes between
systems: same chunks, same questions, same generator, same prompt, same judge.

```
stage 1 (GPU)   colab_retrieve.ipynb          model -> top-5 chunk ids per question            -> Drive: retrieval/<model>.json
stage 2 (API)   colab_generate_judge.ipynb    top-5 chunks -> answer (generator) -> verdict (judge) -> accuracy table
```

Two reference runs bound every result: **no retrieval** (the generator's own knowledge) and **gold chunks**
(retrieval is perfect). The gap between a model and the gold run is what retrieval quality costs; the gap to the
no-retrieval run is what retrieval buys.

## Data and models

- Questions and chunks: RAGTurk `formal_5k` (METU NLP, CC-BY-NC-SA-4.0), the single-file copy on Drive
  (`data/ragturk/ragturk_formal5k.json`). **The data is never committed.** Each question has a reference answer and
  the gold chunk ids. The sample is 500 questions, at most one per article, stratified by question type, seed 42.
- Generator `deepseek-ai/DeepSeek-V4.1-Flash-1017`, judge `zai-org/GLM-5.3-Flash` (different families), both on
  DeepInfra through its OpenAI-compatible API (`https://api.deepinfra.com/v1/openai`).
- Embedding models: our students (folders on Drive) and public baselines at pinned revisions; see the notebook.

## Method notes

- Temperature 0, answers capped at 300 tokens; the judge returns JSON `{verdict, reason}` with verdict in
  `correct | incorrect | abstain`; an unparsable reply is counted as `invalid` and reported, never as wrong.
- Statistics: bootstrap that resamples whole articles; paired differences against a reference system (`*` = the 95%
  interval excludes 0).
- Every API call is cached on disk by the full request, so re-running costs nothing; every command has a hard cap on live
  calls. A cost estimate is printed (prices in `llm.PRICES`, November 2026, the DeepSeek rate is promotional).
- The judge is itself a model: read a sample of its verdicts (the notebook prints five) and hand-check about 30 before
  quoting numbers. RAGTurk questions are LLM-generated and have no human check, so they can share wording with their
  chunks.

## Setup

1. Put the key in Colab Secrets as `DI_TOKEN` (locally: environment variable `DI_TOKEN` or `DEEPINFRA_API_KEY`). It is
   read from the environment only.
2. `pip install -e .` (stage 2) or `pip install -e .[retrieve]` (stage 1). Python >= 3.10.
3. Run `colab_retrieve.ipynb` on a GPU runtime, then `colab_generate_judge.ipynb`. Do the 20-question pilot first.

Command line (stage 2, same code the notebook calls):

```
python -m ragbench.cli smoke
python -m ragbench.cli generate --data D.json --questions Q.json --condition retrieved --retrieval R.json --label NAME
python -m ragbench.cli judge    --data D.json --answers out/answers/NAME__retrieved.jsonl
python -m ragbench.cli report   --data D.json --questions Q.json --reference NAME
```

## Tests

`pytest` (CPU, no network, no API key): the client (cache, retries, call cap), parsing, sampling, top-k search, the
bootstrap, and the whole stage-2 path with a fake transport.
