import os
import json


def read_and_split(fname: str, out_dir: str):
    # Open the original file
    with open(fname, 'r') as original_file:
        lines = original_file.readlines()

    # Calculate partition sizes
    total_lines = len(lines)
    test_size = int(total_lines * 0.1)
    val_size = int(total_lines * 0.1)
    # The rest goes to the train partition

    print(f'There are {total_lines}--> {test_size}, {val_size}, {total_lines - test_size - val_size}')
    print(f'Iterate over {len(lines)} lines in {fname}')

    with open(os.path.join(out_dir, 'data_test.jsonl'), 'w') as test_file, \
         open(os.path.join(out_dir, 'data_val.jsonl'), 'w') as val_file, \
         open(os.path.join(out_dir, 'data_train.jsonl'), 'w') as train_file:

        # Iterate over each line in the original file
        for i, line in enumerate(lines):
            # Parse JSON data (optional, if you need to manipulate the data)
            json_data = json.loads(line)

            # Convert JSON back to string (if manipulated) or use original line
            # json_line = json.dumps(json_data) if 'manipulate' in locals() else line
            # json_line = str(json.dumps(json_data))
            json_line = line

            # Write to appropriate file based on index
            if i < test_size:
                test_file.write(json_line)
            elif i < test_size + val_size:
                val_file.write(json_line)
            else:
                train_file.write(json_line)


if __name__ == "__main__":
    read_and_split('/workspace/data/clean/documents_sft.jsonl', '/workspace/data/')
    print('Done')
