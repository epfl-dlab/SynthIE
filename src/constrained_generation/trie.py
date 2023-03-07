from collections import defaultdict
from typing import List

import jsonlines
import pickle
import os


class Trie(object):
    def __init__(self, sequences: List[List[int]]):
        """Each list in sequences corresponds to a tokenized sequence."""
        next_sets = defaultdict(list)  # a dict that returns an empty list when the key is not in it
        for seq in sequences:
            if len(seq) > 0:
                next_sets[seq[0]].append(seq[1:])

        self._leaves = {k: Trie(v) for k, v in next_sets.items()}
        # for the leaves of the trie _leaves == {}

    def get(self, indices):  # indices holds the list of vocabulary tokens that constitute the current prefix
        if len(indices) == 0:  # if we haven't generated anything so far: return all possible starting tokens
            return list(self._leaves.keys())
        elif indices[0] not in self._leaves:
            # if the currently leading token (and by extension the prefix) isn't eligible: return an empty list
            return []
        else:
            return self._leaves[indices[0]].get(indices[1:])  # continue on the trie corresponding to the leading token

    def dump(self, output_folder_path, file_name, string_iterable=None):
        pickle.dump(self, open(os.path.join(output_folder_path, f"{file_name}.pickle"), "wb"), protocol=4)

        if string_iterable is not None:
            with jsonlines.open(os.path.join(output_folder_path, f"{file_name}_names.jsonl"), "w") as writer:
                writer.write_all(string_iterable)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            trie = pickle.load(f)

        return trie
