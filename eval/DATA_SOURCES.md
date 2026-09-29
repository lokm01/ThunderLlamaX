# Eval data sources (the corpora are NOT published)

The eval harnesses in this directory read corpora from `eval/data/`, which is
intentionally absent from the repository (datasets and book texts are
redistributable-at-best and large). This note is the complete recipe to
recreate them; everything else the battery needs (the PPL code corpora) is
already in this repo.

## GSM8K (gsm8k_eval.py)

OpenAI's Grade-School Math 8K, MIT-licensed: https://github.com/openai/grade-school-math

    mkdir -p eval/data && cd eval/data
    curl -L -o gsm8k_test.jsonl  https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl
    curl -L -o gsm8k_train.jsonl https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl

`gsm8k_eval.py` uses the FIRST 4 train examples as the 4-shot primer and the
FIRST 100 test problems (both deterministic slices of the files above).

## Needle filler (needle_eval.py)

Project Gutenberg texts (public domain in the US; check your jurisdiction):

    curl -L -o gutenberg_pap.txt      https://www.gutenberg.org/files/1342/1342-0.txt   # Pride and Prejudice
    curl -L -o gutenberg_sherlock.txt https://www.gutenberg.org/files/1661/1661-0.txt   # The Adventures of Sherlock Holmes

`needle_eval.py` uses `gutenberg_pap.txt` as filler (the Gutenberg header is
stripped in-code). `gutenberg_sherlock.txt` was staged on the rig as a second
filler option; no committed harness reads it.

## PPL corpora (ppl_moe.py / ppl_dense.py, pre-tokenized by tok_prep.py)

The four scoring domains come from:

| domain         | source text                          | in this repo at                     |
|----------------|--------------------------------------|-------------------------------------|
| `prose`        | Pride and Prejudice (Gutenberg 1342) | `eval/data/gutenberg_pap.txt` (curl above) |
| `code`         | the serving layer                    | `engine/serve.py`                   |
| `prose_private`| the fix-campaign journal             | `docs/history/FIX_CAMPAIGN.md`      |
| `code2`        | the MoE kernel library               | `engine/mm/MM_P7_lib.py`            |

`tok_prep.py` tokenizes each with the model GGUF's tokenizer (asserting the
dense and MoE tokenizers agree) and writes `eval/data/ppl_<domain>_ids.json`.
Re-run it after fetching the texts:

    python3 eval/tok_prep.py

Note the memorization caveat from `P9_EVAL_RESULTS.md`: the Gutenberg classic
is verbatim-memorized by these models (96.5% greedy next-token), so its PPL
1.14 is a CONTAMINATION FLOOR, not a quality number — the honest domains are
`code`, `code2`, and `prose_private`.
