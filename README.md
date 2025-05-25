# open_pref_eval: Fast Preference Evaluation for Local LLMs

**No judge needed.** Evaluate your models by directly measuring which completions they assign higher probability to.

**Why this is better:**
- ⚡ **faster** than judge-based evals (no second model needed)
- 🎯 **More discriminative** with fewer samples (direct probability comparison)
- 🔧 **Actually hackable** (pure PyTorch + HuggingFace, ~2000 lines)

## How it works

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

## Quickstart

```bash
pip install git+https://github.com/wassname/open_pref_eval.git
```

**Simple usage:**
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

See detailed examples: [`examples/example_multiple_models.ipynb`](./examples/example_multiple_models.ipynb)

## Available Datasets

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

**Q: Why is this faster than judge-based evaluation?**  
A: No second model needed. We use your model's own token probabilities, computed in a single forward pass.

**Q: How does this compare to lm-eval-harness or other frameworks?**  
A: I found those hard to get up and running, and to modify. This is pure PyTorch + HuggingFace, with a focus on preference evaluation. You can do lots of quick evals to aid in model development.

**Q: What if my model wasn't trained on preference data?**  
A: Still works! Any model assigns probabilities to text. Well-aligned models tend to assign higher probability to helpful/harmless responses even without explicit preference training.

**Q: Can I add custom datasets?**  
A: Yes! Any dataset with `prompt`, `chosen`, `rejected` columns works. See `examples/` for dataset creation.

# Citing 
If this repository is useful in your own research, you can use the following BibTeX entry:

```
@software{wassname2024reprpo,
  author = {Clark, M.J.},
  title = {open_pref_eval: Fast Preference Evaluation for Local LLMs},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  url = {https://github.com/wassname/open_pref_eval/ },
  commit = {<commit hash>}
}
```
