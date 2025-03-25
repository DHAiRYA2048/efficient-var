#!/usr/bin/env python3
import json
import random
import argparse

def sample_jsonl(input_file, output_file, sample_size=5000):
    """
    Randomly sample entries from a JSONL file.
    
    Args:
        input_file (str): Path to the input JSONL file
        output_file (str): Path to save the sampled output JSONL file
        sample_size (int): Number of entries to sample
    """
    # Read all entries
    entries = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # Skip empty lines
                entries.append(line.strip())
    
    total_entries = len(entries)
    print(f"Read {total_entries} entries from {input_file}")
    
    # Handle case where sample_size is larger than available entries
    if sample_size >= total_entries:
        print(f"Warning: Requested sample size ({sample_size}) is >= total entries ({total_entries})")
        print(f"Using all available entries.")
        sampled_entries = entries
    else:
        # Randomly sample entries
        sampled_entries = random.sample(entries, sample_size)
        print(f"Randomly selected {sample_size} entries")
    
    # Write sampled entries to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in sampled_entries:
            f.write(f"{entry}\n")
    
    print(f"Wrote {len(sampled_entries)} entries to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Randomly sample entries from a JSONL file")
    parser.add_argument("input_file", help="Path to input JSONL file")
    parser.add_argument("output_file", help="Path to output sampled JSONL file")
    parser.add_argument("--sample-size", type=int, default=5000, 
                        help="Number of entries to sample (default: 5000)")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility if provided
    if args.seed is not None:
        random.seed(args.seed)
        print(f"Using random seed: {args.seed}")
    
    sample_jsonl(args.input_file, args.output_file, args.sample_size)
