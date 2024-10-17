from datasets import load_dataset, concatenate_datasets

print("Load Dataset")
dataset = load_dataset("c4", "en", num_proc=128)

print("Shuffle and select data")
sample_data = dataset["train"].shuffle(seed=42).select([i for i in range(120_000_000)])

print("Split pruning set and cpt set")
#Select proportion
num_pruning_sample = 1_200_000
en_pruning_data = sample_data.select([i for i in range(num_pruning_sample)])
en_cpt_data = sample_data.select([i for i in range(num_pruning_sample, len(sample_data))])


zh_dataset = load_dataset("allenai/c4", "zh", num_proc=128, split="train")
zh_dataset = zh_dataset.select([i for i in range(40_000_000)])

num_pruning = 4_000_000
zh_pruning_data = zh_dataset.select([i for i in range(num_pruning)])
zh_cpt_data = zh_dataset.select([i for i in range(num_pruning, len(zh_dataset))])


pruning_data = concatenate_datasets([en_pruning_data, zh_dataset]).shuffle(seed=42)
cpt_data = concatenate_datasets([en_cpt_data, zh_cpt_data]).shuffle(seed=42)

pruning_data.save_to_disk("c4/pruning_raw", num_proc=128)
cpt_data.save_to_disk("c4/cpt", num_proc=128)