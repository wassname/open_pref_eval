# open_pref_eval: Fast Preference Evaluation for Local LLMs

**No judge needed.** Evaluate your models by directly measuring which completions they assign higher probability to.

## Why open_pref_eval?

**Judge-free evaluation.** Unlike other frameworks that rely on LLM judges (GPT-4, Claude) or expensive human annotation, open_pref_eval uses your model's own probabilities.

✅ **No judge bias** - Direct probability measurement vs LLM judge preferences  
✅ **Cost-effective** - Single forward pass vs expensive judge API calls  
✅ **Reproducible** - Deterministic probabilities vs variable judge responses  
✅ **Actually hackable** - Pure PyTorch + HuggingFace, ~2000 lines vs complex frameworks

**Limitations:**
- HuggingFace Transformers models only (extensible)
- Local inference only (extensible for API models)
- Preference datasets only (prompt, chosen, rejected format)
- Results not directly comparable to judge-based frameworks

## How is this different?

| **Approach** | **Judge-based frameworks** | **open_pref_eval** |
|--------------|---------------------------|-------------------|
| **Method** | LLM judges (GPT-4, Claude, etc.) | Direct probability modeling |
| **Cost** | $$ per judge evaluation | $ one-time inference |
| **Bias** | Judge model preferences | Model's own probabilities |
| **Reproducibility** | Variable (judge inconsistency) | Deterministic |
| **Setup** | Complex judge pipelines | Simple: load model + evaluate |

## How It Works

Instead of asking a judge "which is better?", we ask your model "which would you say?":

```python
# Given preference data:
prompt = "Write a helpful response:"
chosen = "I'd be happy to help you with that."
rejected = "I can't assist with anything."

# We measure which completion your model finds more probable:
lp_chosen = model("I'd be happy...").logits.log_softmax(-1).mean()    # -0.31
lp_rejected = model("I can't assist...").logits.log_softmax(-1).mean() # -1.56

# Higher log probability = model prefers it
accuracy = lp_chosen > lp_rejected  # True = model aligned
```

This works because models assign higher probability to completions they prefer. We can directly compare log probabilities without needing a separate judge model.

## 🚀 Quickstart

```bash
pip install git+https://github.com/wassname/open_pref_eval.git
```

> **Evaluating model preferences is as simple as:**

```python
from open_pref_eval.evaluation import evaluate_model
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load your model
model = AutoModelForCausalLM.from_pretrained("your-model")
tokenizer = AutoTokenizer.from_pretrained("your-model")

# Evaluate on multiple datasets
results, raw_data = evaluate_model(
    model=model, 
    tokenizer=tokenizer, 
    datasets=["unalignment/toxic-dpo-v0.2", "wassname/truthful_qa_preferences"]
)

print(results)  # Shows accuracy per dataset
```

**That's it!** No judge setup, no expensive API calls.

**Output:**

| dataset           | correct | prob  | n   | model     |
|:------------------|--------:|------:|----:|:----------|
| toxic-dpo-v0.2    |    0.85 | 0.73  | 300 | your-model|
| truthful_qa       |    0.61 | 0.56  | 300 | your-model|

**Compare multiple models:**
```python
from open_pref_eval import evaluate

# Compare several models at once
results = evaluate(
    model_names=["microsoft/Phi-4-mini-instruct", "Qwen/Qwen3-4B"],
    datasets=["unalignment/toxic-dpo-v0.2"]
)
```

## 💡 Use Cases

- **Model Development**: Compare model versions during training without expensive judges
- **A/B Testing**: Statistically validate preference improvements between model iterations  
- **Research**: Bias-free evaluation for academic papers and experiments
- **Local Evaluation**: Test models offline without API dependencies

See detailed examples: [`examples/example_multiple_models.ipynb`](./examples/example_multiple_models.ipynb)

## ⚡ Key Features

| Feature | open_pref_eval | Judge-based frameworks |
|---------|----------------|------------------------|
| Cost per evaluation | 💚 Single inference | 🟡 Model + judge calls |
| Evaluation bias | 💚 Model's own probs | 🟡 Judge preferences |
| Reproducibility | 💚 Deterministic | 🟡 Variable |
| Setup complexity | 💚 Load model + run | 🟡 Multi-step pipelines |
| Offline capability | 💚 Fully local | 🟡 Often needs APIs |

## 📊 Available Datasets

Built-in datasets covering key safety and capability areas:
- **Safety**: `toxic-dpo-v0.2` (toxicity), `ethics_*` (moral reasoning)  
- **Truthfulness**: `truthful_qa_preferences` (factual accuracy)
- **Helpfulness**: `imdb_preferences` (sentiment), `mmlu_*` (knowledge)

## Technical Details

**Token-level scoring**: We tokenize both completions and compute log probabilities for each token:
```python
# For each token in the completion:
chosen_logprobs = model.logprobs("I'd be happy to help")    # [-0.1, -0.3, -0.2, ...]
rejected_logprobs = model.logprobs("I can't assist")       # [-0.8, -1.2, -0.9, ...]

# Average across tokens (IPO-style):
score = mean(chosen_logprobs) - mean(rejected_logprobs)
accuracy = score > 0  # chosen was more probable
```

**Multiple scoring methods available**: IPO-style (mean), DPO-style (sum), first-diverging token, rank-based, and more in `open_pref_eval.scoring`.

## Why Preference Datasets?

1. **Standardized format** already used in RLHF/DPO training
2. **No judge bias** - your model's own probabilities decide  
3. **Efficient** - single forward pass, no sampling needed
4. **Interpretable** - see exactly which tokens drive the preference

## FAQ

**Q: How does this compare to other evaluation frameworks?**  
A: Most frameworks use LLM judges or human annotation. We use direct probability measurement:
- vs **LM-Eval-Harness**: Focus on preference tasks vs academic benchmarks
- vs **LightEval/Verdict**: No judge setup needed, works with any local model
- vs **Prometheus/HELM**: No proprietary API dependencies or complex infrastructure

**Q: Why is this faster than judge-based evaluation?**  
A: No second model needed. We use your model's own token probabilities, computed in a single forward pass.

**Q: What if my model wasn't trained on preference data?**  
A: Still works! Any model assigns probabilities to text. Well-aligned models tend to assign higher probability to helpful/harmless responses even without explicit preference training.

**Q: Can I add custom datasets?**  
A: Yes! Any dataset with `prompt`, `chosen`, `rejected` columns works. See `examples/` for dataset creation.

## 📚 Documentation

| Section | Description |
|---------|-------------|
| [Examples](./examples/) | Jupyter notebooks with detailed usage |
| [Technical Details](#technical-details) | Scoring methods and implementation |
| [Available Datasets](#-available-datasets) | Built-in preference datasets |
| [FAQ](#faq) | Common questions and comparisons |

## Appendix: Framework Comparison

| Feature | open_pref_eval | [LM-Eval-Harness](https://github.com/EleutherAI/lm-evaluation-harness) | [LightEval](https://github.com/huggingface/lighteval) | [Prometheus](https://github.com/prometheus-eval/prometheus-eval) | [Verdict](https://github.com/haizelabs/verdict) | [HELM](https://github.com/stanford-crfm/helm) |
|---------|----------------|-----------------|-----------|------------|---------|------|
| **Judge Required** | ❌ | ❌ | Optional | ✅ (own models) | ✅ | Optional |



## Citing

If this repository is useful in your own research, you can use the following BibTeX entry:

```
@software{wassname2024open_pref_eval,
  author = {Clark, M.J.},
  title = {open_pref_eval: Fast Preference Evaluation for Local LLMs},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  url = {https://github.com/wassname/open_pref_eval/ },
  commit = {<commit hash>}
}
```
