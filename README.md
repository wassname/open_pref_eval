# OpenPrefEval: Dead Simple Open LLM Evaluation

Tired of spending hours setting up evaluation pipelines? OpenPrefEval lets you benchmark your LLM in 3 lines of code.

```python
from open_pref_eval import evaluate, load_model

results = evaluate(model_name="gpt2", datasets=["unalignment/toxic-dpo-v0.2"])
print(results)
```

Output:


| dataset            |   correct |   prob |   n | model   |
|:-------------------|----------:|-------:|----:|:--------|
| help_steer2-dpo    |      0.39 |  0.486 | 200 | gpt2    |
| toxic-dpo-v0.2     |      1    |  0.715 | 200 | gpt2    |
| truthful_qa_binary |      0.52 |  0.505 | 200 | gpt2    |


![](docs/img/2024-08-03-15-50-51.png)

See more [./examples/evaluate_gpt2.ipynb](./examples/evaluate_gpt2.ipynb)

## Why OpenPreferenceEval?

• Zero setup headaches: Works out-of-the-box with popular open-source LLMs
• Fast: Get results in minutes, not hours
• Flexible: Use our curated datasets or bring your own
• Insightful: Compare your model against common baselines instantly
• Open-source friendly: Designed (ONLY) for models with open weights
* Compatible: built on top of the [transformers](https://github.com/huggingface/transformers) library thus allows the use any model architecture available there.

Stop wrestling with complex eval scripts. Start getting the insights you need, now.

## How it works?

TODO example noteooks

## Status: WIP

- [ ] FIX: is there a length bias?
- [ ] add more datasets
- [ ] make [radar chart](https://matplotlib.org/stable/gallery/specialty_plots/radar_chart.html)
