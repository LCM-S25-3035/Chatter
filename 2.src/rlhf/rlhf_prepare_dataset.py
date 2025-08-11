# rlhf_prepare_dataset.py
from datasets import load_dataset

def main():
    # Load your jsonl file
    dataset = load_dataset('json', data_files=r'C:\Users\sneha\Downloads\projectdsmm\chatter_files\Project\CodesToUpload\S6\rlhf_response_pairs.jsonl', split='train')
    print(dataset)
    
    # Show a few examples
    for example in dataset.select(range(3)):
        print("Prompt:", example['prompt'])
        print("Chosen:", example['chosen'])
        print("Rejected:", example['rejected'])
        print("-"*30)
    
    # Optional: save as HF dataset
    dataset.save_to_disk(r'C:\Users\sneha\Downloads\projectdsmm\chatter_files\Project\CodesToUpload\rlhf\saved_models\rlhf_dataset')

if __name__ == "__main__":
    main()
