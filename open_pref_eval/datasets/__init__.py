from datasets import load_dataset


def get_default_datasets(N):
    dataset_toxic = load_dataset('unalignment/toxic-dpo-v0.2', split=f'train[:{N}]', keep_in_memory=False)

    return [dataset_toxic, 
            load_dataset('wassname/imdb_dpo', name='test', split=f'test[:{N}]', keep_in_memory=False),
            load_dataset('wassname/truthful_qa_dpo', name='validation', split=f'test[:{N}]', keep_in_memory=False),

            load_dataset('wassname/mmlu_dpo', name='elementary_mathematics', split=f'test[:{N}]', keep_in_memory=False), 

            load_dataset('wassname/ethics_expression_dpo', name='commonsense', split=f'test[:{N}]', keep_in_memory=False),
            load_dataset('wassname/ethics_expression_dpo', name='utilitarianism', split=f'test[:{N}]', keep_in_memory=False),
            load_dataset('wassname/ethics_expression_dpo', name='justice', split=f'test[:{N}]', keep_in_memory=False),
            load_dataset('wassname/ethics_expression_dpo', name='deontology', split=f'test[:{N}]', keep_in_memory=False),            
            ]
