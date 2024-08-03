# OpenPrefEval: Dead Simple Open LLM Evaluation

Tired of spending hours setting up evaluation pipelines? OpenPrefEval lets you benchmark your LLM in 3 lines of code.

```python
from openpreferenceeval import evaluate, load_model

model = load_model("your_model_path")
results = evaluate(model, datasets=["HelpSteer2", "trufullqa", "toxic"])
print(results)
```

Output:

| adapter | val_HelpSteer2 | OOD_trufullqa | OOD_toxic |
| :------ | -------------: | ------------: | --------: |
| base    |           0.33 |          0.53 |      0.77 |
| ReprPO  |           0.53 |          0.58 |      0.65 |
| DPO     |           0.68 |          0.73 |      0.45 |

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
