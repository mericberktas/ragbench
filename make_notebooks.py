"""Writes the two thin Colab notebooks into notebooks/. Run once after editing: python make_notebooks.py"""
import json
from pathlib import Path


def cell(kind, text):
    lines = text.strip("\n").split("\n")
    src = [l + "\n" for l in lines[:-1]] + [lines[-1]]
    if kind == "md":
        return {"cell_type": "markdown", "metadata": {}, "source": src}
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src}


def notebook(cells):
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"},
                                         "language_info": {"name": "python"}, "colab": {"provenance": []}},
            "nbformat": 4, "nbformat_minor": 5}


SETUP = '''
import os, sys, json, subprocess
from pathlib import Path
from google.colab import drive

drive.mount('/content/drive')
DRIVE_ROOT = '/content/drive/MyDrive/bitirme'
OUT = f'{DRIVE_ROOT}/ragbench'                       # everything produced goes here (never into the repository)
DATA_PATH = f'{DRIVE_ROOT}/data/ragturk/ragturk_formal5k.json'
REPO = 'https://github.com/mericberktas/ragbench'    # set after the repository is pushed
EXTRA = '{extra}'

# clone under a name that is NOT `ragbench`: the notebook's working directory is /content, and a folder called
# /content/ragbench would shadow the installed package as an empty namespace package
SRC = '/content/ragbench_src'
if not Path(SRC).exists():
    subprocess.run(['git', 'clone', '--depth', '1', REPO, SRC], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-e', SRC + EXTRA], check=True)
import ragbench.data  # fails here, not later, if the install is shadowed
'''

RETRIEVE = [
    cell("md", """
# RAG evaluation, stage 1: retrieval (GPU)

Encodes the RAGTurk chunks and the sampled questions with each model and writes every question's top-5 chunk ids
to Drive (`ragbench/retrieval/<model>.json`). Stage 2 (`colab_generate_judge.ipynb`) needs no GPU: it reads these
files, generates answers with DeepSeek and judges them with GLM. A finished model is skipped on re-run.
"""),
    cell("code", SETUP.replace("{extra}", "[retrieve]")),
    cell("code", """
os.environ['HF_HOME'] = '/content/hf_cache'          # local disk, never Drive
from ragbench.data import load_ragturk, stratified_sample

chunks, all_questions = load_ragturk(DATA_PATH)
Q_PATH = Path(OUT) / 'questions.json'
Q_PATH.parent.mkdir(parents=True, exist_ok=True)
if Q_PATH.exists():
    questions = json.loads(Q_PATH.read_text(encoding='utf-8'))      # the sample is fixed once drawn
else:
    questions = stratified_sample(all_questions, n=500, seed=42)      # at most one question per article
    Q_PATH.write_text(json.dumps(questions, ensure_ascii=False), encoding='utf-8')
print(len(chunks), 'chunks;', len(questions), 'questions')
"""),
    cell("code", """
INSTR_8K = 'Instruct: Given a web search query, retrieve relevant passages that answer the query\\nQuery:'      # no space after "Query:" (model card)
LAB_INSTR = 'Instruct: Given a Turkish search text, retrieve relevant passages written in Turkish that best match the search text.\\nQuery: '
RUN = lambda name: f'{DRIVE_ROOT}/runs/{name}/final'

MODELS = {
    # --- our students ---
    'ours_pair_instr8k_s10800': dict(path=RUN('scale_two_instr8k_s10800_lr1e4/20261027-152516-b2b0c4b')),     # main line (TR-MTEB 67.71)
    'ours_pair_plain_s10800':   dict(path=RUN('scale_two_s10800_lr1e4/20261027-152514-b2b0c4b')),
    'ours_single_bge_s5400':    dict(path=RUN('scale_bge_mix200k_s5400_lr1e4/20261027-101819-77232d7')),
    'ours_single_8k_s5400':     dict(path=RUN('scale_8k_mix200k_s5400_lr1e4/20261027-105836-77232d7')),
    # --- baselines ---
    'raw_base_1k':   dict(hf='ytu-ce-cosmos/modernbert-tr-base-1k', rev='c2e35f553101cce5b03fbb318ef8728473fd34dd'),
    'lab_1k_embed_instr': dict(hf='ytu-ce-cosmos/modernbert-tr-1k-embed', rev='ba1e17bc7e95a257b4e614c300c4ca076711fce6', q=LAB_INSTR),
    'lab_8k_embed_instr': dict(hf='ytu-ce-cosmos/modernbert-tr-embed', rev='855820e0079102805908d60d32cd9fd59c970b80', q=INSTR_8K),
    'bge_m3':        dict(hf='BAAI/bge-m3', rev='5617a9f61b028005a4858fdac845db406aefb181'),
}
for name, spec in MODELS.items():
    if 'path' in spec:
        assert os.path.isdir(spec['path']), f'{name}: {spec["path"]} not found'
print(len(MODELS), 'models')
"""),
    cell("code", """
from ragbench.retrieve import retrieve_for_model

K = 5
for name, spec in MODELS.items():
    out = retrieve_for_model(name, spec, chunks, questions, Path(OUT) / 'retrieval', k=K, doc_format='content',
                             model_dir=f'/content/models/{name}')
    print(f'{name:<30} gold chunk in top-{K}: {out["gold_hit"]:.3f}   ({out["seconds"]} s)')
"""),
    cell("code", """
from google.colab import runtime
runtime.unassign()
"""),
]

GENERATE = [
    cell("md", """
# RAG evaluation, stage 2: answers and judging (no GPU)

For every retrieval file from stage 1 the generator (DeepSeek-V4.1-Flash-1017) answers from the model's top-5
chunks; two reference runs bound the result: **no retrieval** (the generator's own knowledge) and **gold chunks**
(retrieval perfect). The judge (GLM-5.3-Flash, a different model family) compares each answer with the reference
answer. The report gives accuracy per system with article-clustered 95% intervals and paired differences.

The API key is the Colab secret `DI_TOKEN`; calls are cached on Drive (`ragbench/llm_cache.jsonl`), so re-running
a cell costs nothing, and every command has a hard cap on live calls.
"""),
    cell("code", SETUP.replace("{extra}", "")),
    cell("code", """
from google.colab import userdata
os.environ['DI_TOKEN'] = userdata.get('DI_TOKEN')       # never printed, never written to Drive
from ragbench.cli import main
from ragbench.llm import LLMClient
from ragbench.cli import GEN_MODEL, JUDGE_MODEL

# smoke test: 3 live calls per model
main(['smoke', '--n', '3'])
"""),
    cell("code", """
# pilot first: set PILOT to a small number, check a few answers and verdicts by eye, then set it to None
PILOT = 20
Q = f'{OUT}/questions.json'
if PILOT:
    qs = json.loads(Path(Q).read_text(encoding='utf-8'))[:PILOT]
    Q = f'{OUT}/questions_pilot.json'
    Path(Q).write_text(json.dumps(qs, ensure_ascii=False), encoding='utf-8')
    print('pilot of', PILOT, 'questions; the report below needs verdict files for exactly these questions')
OUT_RUN = f'{OUT}/pilot' if PILOT else OUT       # a pilot never mixes with the full run
common = ['--data', DATA_PATH, '--out-dir', OUT_RUN, '--questions', Q, '--max-calls', '4000']
"""),
    cell("code", """
# reference runs, then one run per retrieval file
main(['generate', *common, '--condition', 'none', '--label', 'no_retrieval'])
main(['generate', *common, '--condition', 'gold', '--label', 'gold_chunks'])
for p in sorted(Path(f'{OUT}/retrieval').glob('*.json')):
    main(['generate', *common, '--condition', 'retrieved', '--retrieval', str(p), '--label', p.stem])
"""),
    cell("code", """
for p in sorted(Path(f'{OUT_RUN}/answers').glob('*.jsonl')):
    main(['judge', '--data', DATA_PATH, '--out-dir', OUT_RUN, '--answers', str(p), '--max-calls', '4000'])
"""),
    cell("code", """
main(['report', '--data', DATA_PATH, '--out-dir', OUT_RUN, '--questions', Q, '--reference', 'ours_pair_instr8k_s10800'])
"""),
    cell("code", """
# look at a few judged answers by eye before trusting the numbers
import random
from ragbench.generate import read_jsonl
ref = {q['qid']: q for q in json.loads(Path(Q).read_text(encoding='utf-8'))}
name = 'ours_pair_instr8k_s10800__retrieved'
ans = {a['qid']: a for a in read_jsonl(f'{OUT_RUN}/answers/{name}.jsonl')}
for v in random.Random(0).sample(read_jsonl(f'{OUT_RUN}/verdicts/{name}.jsonl'), 5):
    q = ref[v['qid']]
    print(q['question'], '\\n  reference:', q['answer'][:150], '\\n  answer:   ', ans[v['qid']]['answer'][:150],
          '\\n  verdict:  ', v['verdict'], '-', v['reason'][:120], '\\n')
"""),
]

out = Path(__file__).parent / "notebooks"
out.mkdir(exist_ok=True)
(out / "colab_retrieve.ipynb").write_text(json.dumps(notebook(RETRIEVE), ensure_ascii=False, indent=1), encoding="utf-8")
(out / "colab_generate_judge.ipynb").write_text(json.dumps(notebook(GENERATE), ensure_ascii=False, indent=1), encoding="utf-8")
print("wrote", [p.name for p in out.iterdir()])
