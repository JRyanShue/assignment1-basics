import multiprocessing as mp
import os
import time
from collections import Counter
from typing import BinaryIO

import regex as re


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def pretokenization_worker(filepath: str, start: int, end: int, PAT: str, special_tokens: list[str]) -> Counter:
    """
    Worker for pretokenizing a chunk.

    Args:
        filepath: file to read from.
        start: the start index of the string to process.
        end: the end index.
        PAT: the regex pattern to pretokenize on.
    """
    start_time = time.time()

    with open(filepath, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
    counter = Counter()

    # split across special token (to exclude from pre-tok)
    sub_chunks = re.split("|".join(re.escape(special_token) for special_token in special_tokens), chunk)

    # go through each pretok, adding to counter.
    for sub_chunk in sub_chunks:
        # regex pretok with findall (works because it's a small subchunk)
        counter.update(tuple(char.encode("utf-8") for char in pretok) for pretok in re.findall(PAT, sub_chunk))
        
    print(f"pretokenized first chunk of {len(chunk)} characters in {time.time() - start_time}s.")

    return counter


def pretokenize(num_processes: int, filepath: str, special_tokens: list[str]) -> Counter:
    """

    Returns:
        The pretokenized data, in a Counter of form dict[tuple[bytes], int]. Example element: {(l, o, w): 5}
    """

    assert isinstance(special_tokens, list)
    with open(filepath, "rb") as f:
        if len(special_tokens) == 1:  # jusq1a t endoftext token
            boundaries = find_chunk_boundaries(f, num_processes, special_tokens[0].encode("utf-8"))
        else:
            raise NotImplementedError
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    jobs = [(filepath, start, end, PAT, special_tokens) for start, end in zip(boundaries[:-1], boundaries[1:])]

    # parallelize
    with mp.Pool(processes=num_processes) as pool:
        counters = pool.starmap(pretokenization_worker, jobs)

    # merge
    total = Counter()
    for counter in counters:
        total.update(counter)

    return total


## Usage
if __name__ == "__main__":
    num_processes = 8
    filepath = "/Users/jesseshue/repos/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt"
    special_tokens = ["<|endoftext|>"]

    pretok_counter = pretokenize(num_processes, filepath, special_tokens)
