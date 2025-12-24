import time
from collections import Counter

from cs336_basics.pretokenization_example import pretokenize


def initialize_vocab(special_tokens: list[str]) -> dict:
    """
    Args:
        special_tokens: a list of special tokens, to add to the vocabulary

    Returns:
        A dict of values 0-255 mapping to their corresponding byte value
    """
    # TODO: add special token
    vocab = {}
    vocab_idx = -1
    for vocab_idx in range(256):
        vocab[vocab_idx] = bytes([vocab_idx])

    vocab_idx += 1
    for idx, special_token in enumerate(special_tokens):
        vocab[vocab_idx + idx] = special_token.encode("utf-8")

    return vocab


def initialize_pairs(pretok_counter: Counter) -> Counter:
    """
    Given a vocabulary, store the counts of each unique pair.

    Returns:
        A counter, mapping from a tuple of bytes to its frequency
    """

    pairs_counter = Counter()
    for pretok_bytes, pretok_count in pretok_counter.items():
        pairs = [(pretok_bytes[i], pretok_bytes[i + 1]) for i in range(len(pretok_bytes) - 1)]
        for pair in pairs:
            pairs_counter[pair] += pretok_count

    return pairs_counter


def get_top_pair(pairs_counter: Counter) -> tuple[bytes]:
    """
    Given the pair frequencies, return the highest frequency one, and its frequency. Break ties lexicographically (return higher one).
    """
    highest_freq = max(pairs_counter.values())
    highest_freq_pairs = [pair for pair, freq in pairs_counter.items() if freq == highest_freq]

    if len(highest_freq_pairs) > 1:
        return max(highest_freq_pairs)
    else:
        return highest_freq_pairs[0]


def merge_pretok(pretok: tuple[bytes], pair: tuple[bytes]) -> tuple[tuple[bytes], bytes, tuple[tuple]]:
    """
    Merge items in pretoken given the pair to look for.

    Args:
        pretok: the pretoken to perform merging on
        pair: the pair of tokens within the pretoken to merge

    Returns:
        merged_pretok: a tuple[bytes], the pretoken with instances of pair merged
        new_vocab: a bytes object, the new vocabulary token used to merge
        merged_idxs: the indexes from the original pretoken that were replaced
    """

    matches = []
    for i in range(1 + len(pretok) - 2):
        if pretok[i : i + 2] == pair:
            matches.append((i, i + 2))
    if not matches:
        return None, None, None

    # merge in O(N)
    i, last_j = None, 0
    new_vocab = b"".join(pair)
    pieces = []
    for match in matches:
        i, j = match
        pieces.append(pretok[last_j:i] + (new_vocab,))
        last_j = j
    pieces.append(pretok[last_j:])
    return tuple(x for p in pieces for x in p), new_vocab, tuple(x for i, j in matches for x in (i, j - 1))


def train_bpe_tokenizer(input_path: str, vocab_size: int, special_tokens: list[str]) -> tuple:
    """
    Args:
        input_path: path to a text file with the training data
        vocab_size: maximum final vocab size
        special_tokens: list of strings to add to vocabulary

    Returns:
        vocab: dict[int, bytes]: tokenizer vocabulary, mapping from token id to token bytes
        merges: list[tuple[bytes, bytes]]
    """
    start_time = time.time()

    vocab = initialize_vocab(special_tokens=special_tokens)
    vocab_idx = len(vocab) - 1
    merges = []

    pretok_counter = pretokenize(num_processes=4, filepath=input_path, special_tokens=special_tokens)
    print(f"time to pretokenize: {time.time() - start_time}")
    pairs_counter = initialize_pairs(pretok_counter)

    while len(vocab) < vocab_size:
        highest_freq_pair = get_top_pair(pairs_counter)

        # add to vocab
        vocab[vocab_idx + 1] = b"".join(highest_freq_pair)
        vocab_idx += 1

        # add to merges
        merges.append(highest_freq_pair)

        # update pretok data
        # pair length is always 2.
        for pretok in list(pretok_counter.keys()):  # make static copy to enable popping
            pretok_len = len(pretok)
            if pretok_len < 2:
                continue

            merged_pretok, new_vocab, merged_idxs = merge_pretok(pretok, pair=highest_freq_pair)
            if not merged_pretok:  # match not found
                continue

            # if match: keep frequency for later.
            pretok_freq = pretok_counter.pop(pretok)

            if len(merged_pretok) == 1:
                # there are no new pairs, just a loss of the pair for this pretoken. so rename pretoken and move on.
                pretok_counter[merged_pretok] = pretok_freq
                continue

            # get new pairs (to increment in the pairs counter), and old pairs (to decrement)
            new_vocab_idxs = [i for i, v in enumerate(merged_pretok) if v == new_vocab]
            new_pairs_idxs = set()
            old_pairs_idxs = set()

            # use the indexes, because sometimes a pair occurs multiple times in a pretoken.
            for new_vocab_idx in new_vocab_idxs:
                # attempt to make pairs with the left and right, and remove the old pairs
                if new_vocab_idx > 0:
                    new_pairs_idxs.add((new_vocab_idx - 1, new_vocab_idx))
                    # old_pairs_idxs.add()
                if new_vocab_idx < len(merged_pretok) - 1:
                    new_pairs_idxs.add((new_vocab_idx, new_vocab_idx + 1))

            for i, j in new_pairs_idxs:
                # print(f'adding {pretok_freq} to {(merged_pretok[i], merged_pretok[j])}')
                pairs_counter[(merged_pretok[i], merged_pretok[j])] += pretok_freq

            # old pairs
            for merged_idx in merged_idxs:
                if merged_idx > 0:
                    old_pairs_idxs.add((merged_idx - 1, merged_idx))
                if merged_idx < len(pretok) - 1:
                    old_pairs_idxs.add((merged_idx, merged_idx + 1))

            for i, j in old_pairs_idxs:
                # print(f'subtracting {pretok_freq} from {(pretok[i], pretok[j])}')
                pairs_counter[(pretok[i], pretok[j])] -= pretok_freq

            # rename pretok counter to merged pretok
            pretok_counter[merged_pretok] = pretok_freq

        # pop pair once we're done
        pairs_counter.pop(highest_freq_pair)

    print(f"Total time for BPE training (vocab size {vocab_size}): {time.time() - start_time}")
    # breakpoint()
    return vocab, merges
