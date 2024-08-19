from datasets import load_dataset
from open_pref_eval.datasets.tqa import load_tqa_dpo


def get_default_datasets(N):
    dataset_toxic = load_dataset('unalignment/toxic-dpo-v0.2', split=f'train[:{N}]', keep_in_memory=False)
    # TODO reverse this?

    # dataset_helpsteer2 = load_dataset('Atsunori/HelpSteer2-DPO', split=f'validation[:{N}]', keep_in_memory=False).rename_column('chosen_response', 'chosen').rename_column('rejected_response', 'rejected') # training set

    dataset_tqa = load_tqa_dpo(N) # FIXME load this to hf

    return [dataset_tqa, dataset_toxic, 
            load_dataset('wassname/mmlu_dpo', name='elementary_mathematics', split=f'test[:{N}]', keep_in_memory=False), 

            load_dataset('wassname/ethics_expression_dpo', name='commonsense', split=f'test[:{N}]', keep_in_memory=False),
            load_dataset('wassname/ethics_expression_dpo', name='utilitarianism', split=f'test[:{N}]', keep_in_memory=False),
            load_dataset('wassname/ethics_expression_dpo', name='justice', split=f'test[:{N}]', keep_in_memory=False),
            load_dataset('wassname/ethics_expression_dpo', name='deontology', split=f'test[:{N}]', keep_in_memory=False),            
            ]
