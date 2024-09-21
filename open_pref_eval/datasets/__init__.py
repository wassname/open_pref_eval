from datasets import load_dataset, Dataset
from typing import Optional

def load_dataset_n(path: str, name: Optional[str]=None, split: str = 'test', N:Optional[int] = None) -> Dataset:
    if N is not None:
        split = f'{split}[:{N}]'
    return load_dataset(path, name=name, split=split, keep_in_memory=False)

def ds2name(dataset: Dataset) -> str:
    if getattr(dataset, 'split', None) is None:
        split = ''
    else:
        split = f'-{dataset.split}'

    config_name = dataset.info.config_name
    if config_name == 'default':
        config_name = ''
    else:
        config_name = f'-{config_name}'
    
    return f'{dataset.info.dataset_name}{config_name}{split}'

def get_default_datasets(N=300):
    dataset_toxic = load_dataset_n('unalignment/toxic-dpo-v0.2', split='train', N=N)

    return [dataset_toxic, 
            load_dataset_n('wassname/imdb_preferences', N=N),
            load_dataset_n('wassname/truthful_qa_preferences', split='validation', N=N),
            load_dataset_n('wassname/mmlu_preferences', name='elementary_mathematics', N=N),

            load_dataset_n('wassname/ethics_expression_preferences', name='commonsense', N=N),

            load_dataset_n('wassname/ethics_expression_preferences', name='utilitarianism', N=N),

            load_dataset_n('wassname/ethics_expression_preferences', name='justice', N=N),

            load_dataset_n('wassname/ethics_expression_preferences', name='deontology', N=N),
            ]
