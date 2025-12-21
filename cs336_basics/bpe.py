from pretokenization_example import pretokenize
from collections import Counter


def initialize_vocab() -> dict:
    """
    Returns:
        A dict of values 0-255 mapping to their corresponding byte value
    """
    vocab = {}
    for vocab_idx in range(256):
        vocab[vocab_idx] = bytes([vocab_idx])

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
    Given the pair frequencies, return the highest frequency one. Break ties lexicographically (return higher one).
    """

    highest_freq = max(pairs_counter.values())
    highest_freq_pairs = [pair for pair, freq in pairs_counter.items() if freq == highest_freq]

    if len(highest_freq_pairs) > 1:
        return max(highest_freq_pairs)
    else:
        return highest_freq_pairs[0]


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

    vocab = initialize_vocab()
    vocab_idx = len(vocab) - 1

    pretok_counter = pretokenize(num_processes=8, filepath=input_path, special_tokens=special_tokens)
    pairs_counter = initialize_pairs(pretok_counter)

    while len(vocab) < vocab_size:
        highest_freq_pair = get_top_pair(pairs_counter)

        # add to vocab
        vocab[vocab_idx + 1] = b"".join(highest_freq_pair)
        vocab_idx += 1

        # update pretok data
        # pair length is always 2.
        for pretok in list(pretok_counter.keys()):  # make static copy to enable popping
            pretok_len = len(pretok)
            if pretok_len < 2:
                continue

            # find all matches first, in case there are multiple for one pretoken
            matches = []
            for i in range(1 + pretok_len - 2):
                if pretok[i : i + 2] == highest_freq_pair:
                    matches.append((i, i + 2))
            if not matches:
                continue

            # if match: keep frequency for later.
            pretok_freq = pretok_counter.pop(pretok)

            # merge in O(N)
            i, j, last_j = None, None, 0
            new_vocab = vocab[vocab_idx]
            pieces = []
            for match in matches:
                i, j = match
                pieces.append(pretok[last_j:i] + (new_vocab,))
                last_j = j

                """
                For each changed pair, update pairs counter. 
                When we merge, we get rid of any overlapping pairs.

                We replace them with new pairs containing the new token (if eligible)
                """
                if i > 0:
                    # new pair: the token plus the item on the left
                    # if (pretok[i - 1], new_vocab) in [(b'e', b'ed'), (b'ed', b'e'), (b'ed', b'ed')]:
                    #     breakpoint()

                    if pretok == (b" ", b"n", b"e", b"e", b"d", b"e", b"d"):
                        breakpoint()
                    pairs_counter[(pretok[i - 1], new_vocab)] += pretok_freq
                    print("i > 0")

                    # remove old pairs
                    # if (pretok[i - 1], pretok[i]) == (b'ed', b'ed'):
                    #     breakpoint()
                    pairs_counter[(pretok[i - 1], pretok[i])] -= pretok_freq
                if j < pretok_len:
                    # new pair: the token plus the item on the right
                    # if (new_vocab, pretok[i + 2]) in [(b'e', b'ed'), (b'ed', b'e'), (b'ed', b'ed')]:
                    #     breakpoint()
                    pairs_counter[(new_vocab, pretok[i + 2])] += pretok_freq
                    print("i + 2 < pretok_len")

                    # remove old
                    # if (pretok[i + 2 - 1], pretok[i + 2]) == (b'ed', b'ed'):
                    #     breakpoint()
                    pairs_counter[(pretok[i + 2 - 1], pretok[i + 2])] -= pretok_freq
            # Add end
            pieces.append(pretok[j:])
            merged_pretok = tuple(x for p in pieces for x in p)

            # rename pretok counter to merged pretok
            pretok_counter[merged_pretok] = pretok_freq

            print(f"Matched {highest_freq_pair} in {pretok}. Turned it into {merged_pretok}.")

            if pretok == (b" ", b"c", b"h", b"e", b"r", b"i", b"s", b"h", b"e", b"d"):
                print("yeuye")
                breakpoint()

        # pop pair once we're done

        # breakpoint()
        pairs_counter.pop(highest_freq_pair)

        # Update pairs

        # breakpoint()

    breakpoint()
