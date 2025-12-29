import time
from collections import Counter, defaultdict
from curses import pair_content

from sympy.assumptions.assume import true
from sympy.printing.pretty.pretty_symbology import B
from tqdm import tqdm

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


def initialize_pairs(pretok_counter: Counter) -> tuple[Counter, dict]:
    """
    Given a vocabulary, store the counts of each unique pair.

    Returns:
        A counter, mapping from a tuple of bytes to its frequency
    """

    pairs_counter = Counter()
    pairs_to_pretoks = defaultdict(set)
    for pretok_bytes, pretok_count in pretok_counter.items():
        pairs = [(pretok_bytes[i], pretok_bytes[i + 1]) for i in range(len(pretok_bytes) - 1)]
        for pair in pairs:
            pairs_counter[pair] += pretok_count
            pairs_to_pretoks[pair].add(pretok_bytes)

    return pairs_counter, pairs_to_pretoks


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
    Merge items in pretoken given the pair to look for. Non-overlapping merges.

    Args:
        pretok: the pretoken to perform merging on
        pair: the pair of tokens within the pretoken to merge

    Returns:
        merged_pretok: a tuple[bytes], the pretoken with instances of pair merged
        new_vocab: a bytes object, the new vocabulary token used to merge
        merged_idxs: the indexes from the original pretoken that were replaced
    """

    # what chatty recommended
    a, b = pair

    # shortcut
    if a not in pretok or b not in pretok:
        return None, None, None

    out = []
    merged_idxs = []
    i = 0
    n = len(pretok)
    did_merge = False
    new_vocab = a + b

    while i < n:
        if i + 1 < n and pretok[i] == a and pretok[i + 1] == b:
            out.append(new_vocab)
            merged_idxs.extend([i, i + 1])
            did_merge = True
            i += 2
        else:
            out.append(pretok[i])
            i += 1

    if not did_merge:
        return None, None, None

    return tuple(out), new_vocab, tuple(merged_idxs)

    # # mine
    # # BIG BUG: by looking for matches first, something like "sss" -> "ss", "ss"
    # a, b = pair

    # # shortcut
    # if a not in pretok or b not in pretok:
    #     return None, None, None

    # matches = []
    # for i in range(1 + len(pretok) - 2):
    #     if pretok[i : i + 2] == pair:
    #         matches.append((i, i + 2))
    # if not matches:
    #     return None, None, None

    # # merge in O(N)
    # i, last_j = None, 0
    # new_vocab = b"".join(pair)
    # pieces = []
    # for match in matches:
    #     i, j = match
    #     pieces.append(pretok[last_j:i] + (new_vocab,))
    #     last_j = j
    # pieces.append(pretok[last_j:])

    # return tuple(x for p in pieces for x in p), new_vocab, tuple(x for i, j in matches for x in (i, j - 1))

    # chatty
    # a, b = pair
    # new_vocab = a + b

    # out = []
    # merged_idxs = []

    # n = len(pretok)
    # i = 0
    # while i < n:
    #     if i + 1 < n and pretok[i] == a and pretok[i + 1] == b:
    #         out.append(new_vocab)
    #         merged_idxs.append(i)
    #         merged_idxs.append(i + 1)
    #         i += 2
    #     else:
    #         out.append(pretok[i])
    #         i += 1

    # if not merged_idxs:
    #     return None, None, None
    # return tuple(out), new_vocab, tuple(merged_idxs)


def initialize_bpe_tokenizer(input_path: str, special_tokens: list[str]) -> tuple[Counter, Counter, dict]:
    """
    Initializes the two central elements to BPE training: the pretok_counter and the pairs_counter
    """

    pretok_counter = pretokenize(num_processes=4, filepath=input_path, special_tokens=special_tokens)
    pairs_counter, pairs_to_pretoks = initialize_pairs(pretok_counter)

    return pretok_counter, pairs_counter, pairs_to_pretoks


def train_bpe_tokenizer(
    vocab_size: int, special_tokens: list[str], pretok_counter: Counter, pairs_counter: Counter, pairs_to_pretoks: dict
) -> tuple:
    """
    Args:
        input_path: path to a text file with the training data
        vocab_size: maximum final vocab size
        special_tokens: list of strings to add to vocabulary

    Returns:
        vocab: dict[int, bytes]: tokenizer vocabulary, mapping from token id to token bytes
        merges: list[tuple[bytes, bytes]]
    """

    print("\n\n")

    vocab = initialize_vocab(special_tokens=special_tokens)
    vocab_idx = len(vocab) - 1
    merges = []

    pbar = tqdm(total=vocab_size, initial=len(vocab))

    total_pretok_checking_time = 0
    total_get_top_pair_time = 0
    start_time = time.perf_counter()

    while len(vocab) < vocab_size:
        last_time = time.perf_counter()
        highest_freq_pair = get_top_pair(pairs_counter)
        total_get_top_pair_time += time.perf_counter() - last_time

        # add to vocab
        vocab[vocab_idx + 1] = b"".join(highest_freq_pair)
        vocab_idx += 1

        # add to merges
        merges.append(highest_freq_pair)

        # update pretok data
        # pair length is always 2.
        last_time = time.perf_counter()

        # if len(b"".join(highest_freq_pair)) > 2:
        #     breakpoint()

        # TODO: Implement
        # pretoks_to_skip = set()

        # Merge the relevant pretokens, updating the pairs_counter and pairs_to_pretokens
        # print("NEW")
        pretoks_to_check = pairs_to_pretoks[highest_freq_pair].copy()  # NEED TO UPDATE THIS
        while len(pretoks_to_check) > 0:
            
            pretok = pretoks_to_check.pop()  # Order of checking doesn't matter, so long as everything is valid.  
            # print(f"popped from pretoks_to_check: {pretok}")

            # if pretok not in pairs_to_pretoks[highest_freq_pair]:
            #     continue

            # if pretok in pretoks_to_skip:
            #     continue

        # for pretok in list(pretok_counter.keys()):  # make static copy to enable popping
            pretok_len = len(pretok)
            if pretok_len < 2:
                continue

            merged_pretok, new_vocab, merged_idxs = merge_pretok(pretok, pair=highest_freq_pair)

            if not merged_pretok:  # match not found
                continue

            # if match: keep frequency for later.
            # TODO: BUG BUG BUG
            if pretok not in pretok_counter:
                breakpoint()
            # if pretok == (b' ', b't', b'o', b'u', b'g', b'h', b'e', b'r'):
            #     breakpoint()
            # print(f"popping from pretok_counter {pretok}")
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
                pairs_to_pretoks[(merged_pretok[i], merged_pretok[j])].add(merged_pretok)  # NEED MORE
                if (merged_pretok[i], merged_pretok[j]) == highest_freq_pair:
                    # print(f"adding to pretoks_to_check: {merged_pretok}")
                    pretoks_to_check.add(merged_pretok)


            # add pretok to pairs in it
            # an update to pairs_to_pretoks for the pretoks that have been newly created (via merging)
            for i in range(len(merged_pretok) - 1):
                j = i + 1
                pairs_to_pretoks[(merged_pretok[i], merged_pretok[j])].add(merged_pretok)
                # if (i, j) == highest_freq_pair:
                #     print(f"{(i, j)} in {merged_pretok}. need to add")


            # old pairs
            for merged_idx in merged_idxs:
                if merged_idx > 0:
                    old_pairs_idxs.add((merged_idx - 1, merged_idx))
                if merged_idx < len(pretok) - 1:
                    old_pairs_idxs.add((merged_idx, merged_idx + 1))

            for i, j in old_pairs_idxs:
                # print(f'subtracting {pretok_freq} from {(pretok[i], pretok[j])}')
                
                pairs_counter[(pretok[i], pretok[j])] -= pretok_freq
                # print(f"removing {pretok} from old pair {(pretok[i], pretok[j])}")


            # check all pretok pairs to see which pairs to remove pretok from 
            # basically a check to see which pretoks don't exist (because they've been merged)
            for i in range(len(pretok) - 1):
                j = i + 1
                if pretok in pairs_to_pretoks[(pretok[i], pretok[j])]:
                    pairs_to_pretoks[(pretok[i], pretok[j])].remove(pretok)
                # if (i, j) == highest_freq_pair:
                #     print(f"{(i, j)} in {pretok}. need to remove")
                        # if pretok == (b' ', b't', b'h', b'e', b'm'):
                        #     breakpoint()
                        # if (pretok[i], pretok[j]) == highest_freq_pair:
                        #     if pretok not in pretoks_to_check:
                        #         breakpoint()
                        #     print(f"removing {pretok} from pretoks_to_check")
                        #     pretoks_to_check.remove(pretok)

            # rename pretok counter to merged pretok
            pretok_counter[merged_pretok] = pretok_freq

        total_pretok_checking_time += time.perf_counter() - last_time

        # pop pair once we're done
        pairs_counter.pop(highest_freq_pair)
        pairs_to_pretoks.pop(highest_freq_pair)

        # update pbar
        # print(f"Added {b''.join(highest_freq_pair)} to the vocab.")
        pbar.update(len(vocab) - pbar.n)

    pbar.close()

    total_time = time.perf_counter() - start_time
    print(f"Total time to train: {total_time}")
    print(
        f"Total time cycling through pretoks: {total_pretok_checking_time} ({100 * total_pretok_checking_time / total_time}%)"
    )
    print(f"Total time getting top pairs: {total_get_top_pair_time} ({100 * total_get_top_pair_time / total_time}%)")

    return vocab, merges


if __name__ == "__main__":
    special_tokens = ["<|endoftext|>"]

    start_time = time.time()
    pretok_counter, pairs_counter, pairs_to_pretoks = initialize_bpe_tokenizer(
        input_path="/Users/jesseshue/repos/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt",
        special_tokens=special_tokens,
    )
    last_time = time.time()
    total_init_time = last_time - start_time
    print(f"time to pretokenize and initialize pairs: {total_init_time}")

    vocab, merges = train_bpe_tokenizer(
        vocab_size=10000,
        special_tokens=special_tokens,
        pretok_counter=pretok_counter,
        pairs_counter=pairs_counter,
        pairs_to_pretoks=pairs_to_pretoks,
    )
