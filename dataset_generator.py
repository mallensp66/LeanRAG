import os
import json
import hashlib
from pathlib import Path

# --- Configuration ---
SOURCE_DIRECTORY = './datasets/five'  # Replace with your directory path
OUTPUT_FILE = f"{SOURCE_DIRECTORY}/input.jsonl"

def generate_unique_id(content):
    """Generates a 32-character hex ID based on the content (MD5)."""
    return hashlib.md5(content.encode('utf-8')).hexdigest()

def process_files(source_dir, output_file):
    source_path = Path(source_dir)
    
    # Check if directory exists
    if not source_path.exists():
        print(f"Error: Directory '{source_dir}' not found.")
        return

    print(f"Processing files from: {source_path.resolve()}")
    
    with open(output_file, 'w', encoding='utf-8') as outfile:
        count = 0
        
        # Iterate over all files in the directory
        for file_path in source_path.glob('*.md'):
            try:
                # Read the markdown content
                with open(file_path, 'r', encoding='utf-8') as f:
                    context_content = f.read()

                # Calculate length of the content
                content_length = len(context_content)
                
                # Generate a unique ID (using hash of content for consistency)
                unique_id = generate_unique_id(context_content)

                # Construct the data object
                data = {
                    "input": "What is the main topic of this document?", # Placeholder question
                    "context": context_content,
                    "dataset": "dummy",
                    "label": "longbench",
                    "answers": ["dummy answer"],
                    "_id": unique_id,
                    "length": content_length
                }

                # Write to jsonl (one JSON object per line)
                json_line = json.dumps(data, ensure_ascii=False)
                outfile.write(json_line + '\n')
                
                count += 1
                
            except Exception as e:
                print(f"Skipping {file_path.name}: {e}")

    print(f"---")
    print(f"Success! Processed {count} files.")
    print(f"Output saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    # Create a dummy directory/file for testing purposes if you run this immediately
    if not os.path.exists(SOURCE_DIRECTORY):
        os.makedirs(SOURCE_DIRECTORY)
        with open(os.path.join(SOURCE_DIRECTORY, "test_file.md"), "w") as f:
            f.write("# Hello World\nThis is a test markdown file content.")
            
    process_files(SOURCE_DIRECTORY, OUTPUT_FILE)