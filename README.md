# Backdoor LLM Tester

A small, notebook-first workflow for comparing a backdoored Hugging Face instruction/chat model with a clean checkpoint of the same backbone.

## Quick start

1. Open `backdoor_evaluation.ipynb` on a GPU server.
2. On Kaggle, try the imports first; run the dependency cell only if a package is missing. Do not replace Kaggle's preinstalled PyTorch.
3. Edit the **Run configuration** cell for both models, datasets, and server.
4. Run the notebook from top to bottom.

The notebook downloads revision-pinned snapshots and lets Transformers/Accelerate distribute each model over every visible GPU. To keep VRAM use practical, it evaluates and unloads the backdoored model before loading the clean model.

Both checkpoints are evaluated on exactly the same inputs:

- paired clean/triggered conversations for backdoor success and false activation;
- WikiText language-modeling text for token-weighted perplexity;
- GSM8K reasoning questions for exact final-answer accuracy.

The checkpoints must have the same core architecture. Set `CLEAN_MODEL_ID` to the known-clean instruction/chat checkpoint corresponding to `BACKDOORED_MODEL_ID`.

Use `CUDA_VISIBLE_DEVICES` before starting Jupyter to select GPUs. For example:

```bash
CUDA_VISIBLE_DEVICES=0,1 jupyter lab
```

The included [`data/example_backdoor_tests.jsonl`](data/example_backdoor_tests.jsonl) shows the input schema. Replace its placeholder targets with the known trigger and expected behavior for the model under test.

## GPU sizing

- Start with `DTYPE = "auto"`, `LOAD_IN_4BIT = False`, and `LOAD_IN_8BIT = False`.
- On smaller GPUs such as T4s, set `DTYPE = "float16"` and enable 4-bit loading if the unquantized model does not fit.
- On an H200 or another BF16-capable GPU, `DTYPE = "bfloat16"` is usually appropriate.
- `device_map="auto"` uses all GPUs visible to the notebook. `GPU_MEMORY_RESERVE_GB` leaves headroom on each device.
- Quantization and CPU offload are never enabled automatically; all memory/quality tradeoffs remain explicit in the configuration cell.

## JSONL fields

Each non-empty line is one paired test:

- `id`: unique stable identifier
- `clean_messages`: normal chat messages
- `triggered_messages`: the matched conversation containing the known trigger
- `target`: expected malicious/backdoor response text or regex
- `target_match_type`: `exact`, `contains`, or `regex`
- `clean_expected`: optional expected clean response
- `clean_match_type`: optional matcher for `clean_expected` (defaults to `contains`)

The notebook treats ASR as target activation on triggered inputs. It also reports target activation on clean inputs, trigger uplift, conditional flip rate, and—where `clean_expected` is supplied—clean accuracy.

## Clean evaluation

The defaults are `Salesforce/wikitext` (`wikitext-2-raw-v1`) and `openai/gsm8k` (`main`). The sample and token limits in the configuration cell support quick runs; increase them for final measurements.

Perplexity uses token-weighted causal cross-entropy with overlapping context windows. GSM8K scoring extracts and exactly compares the final numeric answer. `comparison.json` and `comparison.csv` report both checkpoint values and `backdoored - clean` deltas.

Dataset downloads require internet access on the first run and are cached by Hugging Face for later reuse.
