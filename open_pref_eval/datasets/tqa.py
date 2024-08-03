from datasets import load_dataset

def load_tqa_dpo(N=None):

    slice = '' if N is None else f'[:{N}]' # https://huggingface.co/docs/datasets/en/loading#slice-splits
    dataset_tqab = load_dataset("EleutherAI/truthful_qa_binary", split=f'validation{slice}', keep_in_memory=False)
    def proc(row):
        l = row['label']
        return {
            'prompt': row['question'],
            'chosen': row['choices'][l],
            'rejected': row['choices'][~l],
        }
    return dataset_tqab.map(proc).select_columns(['prompt', 'chosen', 'rejected'])
