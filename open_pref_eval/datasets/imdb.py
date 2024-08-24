from datasets import load_dataset

def proc(row):
    l = row['label']
    prompt = f"""Question: Does the following movie review have positive or negative in sentiment?

Review: {row['text']}

Answer: The sentiment of the above review is"""
    # 0 is negative, 1 is positive
    chosen = 'positive' if l == 1 else 'negative'
    rejected = 'positive' if l == 0 else 'negative'

    return {
        'prompt': prompt,
        'chosen': chosen,
        'rejected': rejected,
    }

def load_imdb_dpo(N=None):

    slice = '' if N is None else f'[:{N}]' # https://huggingface.co/docs/datasets/en/loading#slice-splits
    # Warning this is not shuffled
    dataset = load_dataset("stanfordnlp/imdb", split=f'test{slice}', keep_in_memory=False)
    return dataset.map(proc).select_columns(['prompt', 'chosen', 'rejected'])
