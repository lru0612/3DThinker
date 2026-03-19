import json

# Read JSONL file content
file_path = "../../data/example.jsonl"  # Replace with your JSONL file path
output_file_path = "../../data/example.json"  # Your output JSON file path

formatted_data = []

# Read and process each line of data
with open(file_path, 'r') as f:
    for idx, line in enumerate(f):
        data = json.loads(line.strip())

        # Format into the specified JSON structure
        formatted_item = {
            "idx": data["idx"],
            "dataset": "mindcube",
            "problem": data["text_input"],
            "response": data["answer"],
            "reasoning": data["text_output"],
            "images": data["image_input"],
            "images_type": "full"
        }

        formatted_data.append(formatted_item)

# Save the formatted content to a JSON file
with open(output_file_path, 'w') as of:
    json.dump(formatted_data, of, ensure_ascii=False, indent=4)

print(f"Formatted data saved to {output_file_path}")
