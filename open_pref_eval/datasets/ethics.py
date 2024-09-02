from datasets import load_dataset

def get_ethics_datasets(N=None):

    def load_data_with_N(ds, name, split, N=None):
        if N is not None:
            split += f'[:{N}]'
        return load_dataset(ds, name=name, split=split, keep_in_memory=False)

    return [
        load_data_with_N('wassname/ethics_expression_dpo', name='commonsense', split='test', N=N),
        load_data_with_N('wassname/ethics_expression_dpo', name='utilitarianism', split='test', N=N),
        load_data_with_N('wassname/ethics_expression_dpo', name='justice', split='test', N=N),
        load_data_with_N('wassname/ethics_expression_dpo', name='deontology', split='test', N=N),
    ]
