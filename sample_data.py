import random
import itertools

def sample_lines_from_file(input_file, output_file, sample_size=100, start_line=0, lines_to_read=500, seed=42):
    """
    Randomly sample lines from a specified portion of a file.

    :param input_file: Path to the source file to sample from
    :param output_file: Path to the destination file for sampled lines
    :param sample_size: Number of lines to sample (default 100)
    :param start_line: Line number to start reading from (default 0, first line)
    :param lines_to_read: Number of lines to read after start_line (default 500)
    :param seed: Random seed for reproducibility (default 42)
    """
    # Set the random seed for reproducibility
    random.seed(seed)

    try:
        # Open the file and skip to the start line
        with open(input_file, 'r', encoding='utf-8') as f:
            # Skip lines before start_line
            for _ in range(start_line):
                next(f)

            # Read the specified number of lines
            lines = list(itertools.islice(f, lines_to_read))

    except StopIteration:
        # If file has fewer lines than expected
        print(f"Warning: File has fewer lines than expected starting from line {start_line}")
        lines = []
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        return
    except IOError as e:
        print(f"Error reading input file: {e}")
        return

    # Randomly sample lines
    try:
        # Ensure we don't try to sample more lines than are available
        sampled_lines = random.sample(lines, min(sample_size, len(lines)))

        # Write sampled lines to output file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.writelines(sampled_lines)

        print(f"Successfully sampled {len(sampled_lines)} lines to {output_file}")

    except ValueError as e:
        print(f"Error sampling lines: {e}")
    except IOError as e:
        print(f"Error writing to output file: {e}")

# Example usage
if __name__ == "__main__":
    input_filename = "dataset/riddlesense/rs_dev.jsonl"
    output_filename = "dataset/riddlesense/rs_test_targets.jsonl"
    # output_filename = "dataset/riddlesense/rs_test_examples.jsonl"

    # Example: Sample 100 lines starting from line 500 (501st line to 1000th line)
    sample_lines_from_file(
        input_filename,
        output_filename,
        sample_size=100,  # Number of lines to sample
        start_line=0,   # Start reading from the 501st line
        lines_to_read=500 # Read 500 lines from the start line
    )
