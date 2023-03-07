import unicodedata
import re

from abc import ABC
from typing import List, Tuple

from src import utils

log = utils.get_pylogger(__name__)


def get_linearization_class(identifier: str):
    supported_linearization_classes = [
        FullyExpandedLinearization,
        FullyExpandedLinearizationET,
        SubjectCollapsedLinearization,
    ]

    for linearization_class in supported_linearization_classes:
        if linearization_class.identifier == identifier:
            return linearization_class

    raise ValueError("Unknown linearization class identifier: {}".format(identifier))


class LinearizationType(ABC):
    @staticmethod
    def process_triplet_text_parts(text_parts, return_set, verbose, text=""):
        if verbose and len(text_parts) % 3 != 0:
            # log.info(
            #     f"Textual sequence: ```{text}``` does not follow the {cls.subject_id}, {cls.relation_id},"
            #     f" {cls.object_id}, {cls.et_id} format!")
            log.info(f"Textual sequence: ```{text}``` does not follow the triplet format!")

        text_triplets = [tuple(text_parts[i : i + 3]) for i in range(0, len(text_parts) - 2, 3)]

        if not return_set:
            return text_triplets

        unique_text_triplets = set(text_triplets)

        if verbose and len(unique_text_triplets) != len(text_triplets):
            log.info(f"Textual sequence: ```{text}``` has duplicated triplets!")

        return unique_text_triplets

    @staticmethod
    def normalize_spaces(name, keep_spaces: bool) -> str:
        name = name.strip()
        if keep_spaces:
            return name.replace("_", " ")

        return name.replace(" ", "_")

    @classmethod
    def triplet_list_to_text(cls, **kwargs):
        raise NotImplementedError()

    @classmethod
    def text_to_triplet_list(cls, **kwargs):
        raise NotImplementedError()


class FullyExpandedLinearization(LinearizationType):
    identifier = "fully_expanded"
    subject_id = "[s]"
    relation_id = "[r]"
    object_id = "[o]"
    et_id = "[e]"
    separator = " "
    keep_spaces_entities = False
    keep_spaces_relations = True
    triplet_format_parts = [
        "{subject_id}",
        " ",
        "{subject}",
        " ",
        "{relation_id}",
        " ",
        "{relation}",
        " ",
        "{object_id}",
        " ",
        "{object}",
        " ",
        "{et_id}",
    ]

    @classmethod
    def triplet_list_to_text(
        cls,
        triplet_list: List[Tuple[str, str, str]],
        tokenizer=None,
    ):
        linearized_triplets_text_parts = []

        surface_forms = {
            "{subject_id}": cls.subject_id,
            "{relation_id}": cls.relation_id,
            "{object_id}": cls.object_id,
            "{et_id}": cls.et_id,
            "{separator}": cls.separator,
        }

        for sub, rel, obj in triplet_list:
            # Complete the necessary surface forms for the current triplet and process them
            surface_forms.update({"{subject}": sub, "{relation}": rel, "{object}": obj})
            surface_forms["{subject}"] = cls.normalize_spaces(surface_forms["{subject}"], cls.keep_spaces_entities)
            surface_forms["{relation}"] = cls.normalize_spaces(surface_forms["{relation}"], cls.keep_spaces_relations)
            surface_forms["{object}"] = cls.normalize_spaces(surface_forms["{object}"], cls.keep_spaces_entities)

            # Format the triplet and add its constituent parts to the list of linearized triplets
            linearized_triplets_text_parts.extend([surface_forms.get(item, item) for item in cls.triplet_format_parts])
            linearized_triplets_text_parts.append(surface_forms["{separator}"])

        linearized_triplets_text_parts = linearized_triplets_text_parts[:-1]
        text = "".join(linearized_triplets_text_parts)

        if tokenizer is None:
            # If tokenizer has not been provided, return only the text corresponding to the linearized triplet
            return text, None

        # Tokenize each part separately to ensure that the tokenization is consistent with the constrained decoding
        tokenized_linearized_triplets_text_parts = tokenizer(
            linearized_triplets_text_parts, return_attention_mask=False
        )["input_ids"]
        # Concatenate the tokenized parts to obtain the tokenized linearized triplet
        tokenized_text = [token for part in tokenized_linearized_triplets_text_parts for token in part[:-1]] + [
            tokenizer.eos_token_id
        ]

        # If tokenizer has been provided, return both the text and the ids corresponding to the linearized triplet
        return text, tokenized_text

    @classmethod
    def _split_text(cls, text):
        surface_forms = {
            "{subject_id}": cls.subject_id,
            "{relation_id}": cls.relation_id,
            "{object_id}": cls.object_id,
            "{et_id}": cls.et_id,
            "{separator}": cls.separator,
        }

        regex = re.compile(
            "|".join(
                [re.escape(surface_forms[_id]) for _id in ["{subject_id}", "{relation_id}", "{object_id}", "{et_id}"]]
            )
        )
        text_parts = [
            element.strip(surface_forms["{separator}"]).strip().strip(surface_forms["{separator}"])
            for element in re.split(regex, text)
            if element.strip()
        ]

        return text_parts

    @classmethod
    def text_to_triplet_list(cls, text, verbose=True, return_set=True):
        """
        Maps a model's output string to the corresponding triplets.
        Must be consistent with TripletUtils.triplets_to_target_format().

        :param text: model's output string
        :param verbose: flag to print debug messages (e.g., incorrect formatting, duplicated triplets)
        :param return_set: flag indicating whether to return the triplets as a set (and filter out potential duplicates)
        :return: a list or a set of the triplets in the model's output string
        """
        text_parts = cls._split_text(text)

        if verbose and len(text_parts) % 3 != 0:
            # log.info(
            #     f"Textual sequence: ```{text}``` does not follow the {cls.subject_id}, {cls.relation_id},"
            #     f" {cls.object_id}, {cls.et_id} format!")
            log.info(
                f"Textual sequence: ```{text}``` does not follow the {cls.subject_id}, {cls.relation_id},"
                f" {cls.object_id}, {cls.et_id} format!"
            )

        text_triplets = [tuple(text_parts[i : i + 3]) for i in range(0, len(text_parts) - 2, 3)]

        if not return_set:
            return text_triplets

        unique_text_triplets = set(text_triplets)

        if verbose and len(unique_text_triplets) != len(text_triplets):
            log.info(f"Textual sequence: ```{text}``` has duplicated triplets!")

        return unique_text_triplets


class FullyExpandedLinearizationET(FullyExpandedLinearization):
    identifier = "fully_expanded_et"
    et_id = "[et]"


class SubjectCollapsedLinearization(FullyExpandedLinearization):
    identifier = "subject_collapsed"
    subject_id = "[s]"
    relation_id = "[r]"
    object_id = "[o]"
    et_id = "[e]"
    separator = " "
    keep_spaces_entities = False
    keep_spaces_relations = True
    inside_triplet_format_parts = ["{relation_id}", " ", "{relation}", " ", "{object_id}", " ", "{object}", " "]
    starting_triplet_format_parts = [
        "{et_id}",
        "{separator}",
        "{subject_id}",
        " ",
        "{subject}",
        " ",
        "{relation_id}",
        " ",
        "{relation}",
        " ",
        "{object_id}",
        " ",
        "{object}",
        " ",
    ]

    @classmethod
    def triplet_list_to_text(
        cls,
        triplet_list: List[Tuple[str, str, str]],
        tokenizer=None,
    ):
        linearized_triplets_text_parts = []

        surface_forms = {
            "{subject_id}": cls.subject_id,
            "{relation_id}": cls.relation_id,
            "{object_id}": cls.object_id,
            "{et_id}": cls.et_id,
            "{separator}": cls.separator,
        }

        current_sub = None
        for sub, rel, obj in triplet_list:
            surface_forms.update({"{subject}": sub, "{relation}": rel, "{object}": obj})
            surface_forms["{subject}"] = cls.normalize_spaces(surface_forms["{subject}"], cls.keep_spaces_entities)
            surface_forms["{relation}"] = cls.normalize_spaces(surface_forms["{relation}"], cls.keep_spaces_relations)
            surface_forms["{object}"] = cls.normalize_spaces(surface_forms["{object}"], cls.keep_spaces_entities)

            if current_sub != sub or current_sub is None:
                current_sub = sub
                triplet_format_parts = cls.starting_triplet_format_parts
            else:
                triplet_format_parts = cls.inside_triplet_format_parts

            # Format the triplet and add its constituent parts to the list of linearized triplets
            linearized_triplets_text_parts.extend([surface_forms.get(item, item) for item in triplet_format_parts])

        linearized_triplets_text_parts = linearized_triplets_text_parts[2:]
        linearized_triplets_text_parts.append(surface_forms["{et_id}"])
        text = "".join(linearized_triplets_text_parts)

        if tokenizer is None:
            # If tokenizer has not been provided, return only the text corresponding to the linearized triplet
            return text, None

        # Tokenize each part separately to ensure that the tokenization is consistent with the constrained decoding
        tokenized_linearized_triplets_text_parts = tokenizer(
            linearized_triplets_text_parts, return_attention_mask=False
        )["input_ids"]
        # Concatenate the tokenized parts to obtain the tokenized linearized triplet
        tokenized_text = [token for part in tokenized_linearized_triplets_text_parts for token in part[:-1]] + [
            tokenizer.eos_token_id
        ]

        # If tokenizer has been provided, return both the text and the ids corresponding to the linearized triplet
        return text, tokenized_text

    @classmethod
    def _split_text(cls, text, flatten=False):
        surface_forms = {
            "{subject_id}": cls.subject_id,
            "{relation_id}": cls.relation_id,
            "{object_id}": cls.object_id,
            "{et_id}": cls.et_id,
            "{separator}": cls.separator,
        }

        regex = re.compile(
            "|".join([re.escape(surface_forms[_id]) for _id in ["{subject_id}", "{relation_id}", "{object_id}"]])
        )

        collapsed_triplets = [triplet_text.strip(cls.separator) for triplet_text in text.split(cls.et_id)]

        collapsed_triplets_parts = [
            [
                element.strip(surface_forms["{separator}"]).strip().strip(surface_forms["{separator}"])
                for element in re.split(regex, triplet_text)
                if element.strip()
            ]
            for triplet_text in collapsed_triplets
            if triplet_text.strip()
        ]

        if flatten:
            return [element for triplet in collapsed_triplets_parts for element in triplet]

        return collapsed_triplets_parts

    @classmethod
    def text_to_triplet_list(cls, text, verbose=True, return_set=True):
        """
        Maps a model's output string to the corresponding triplets.
        Must be consistent with TripletUtils.triplets_to_target_format().

        :param text: model's output string
        :param verbose: flag to print debug messages (e.g., incorrect formatting, duplicated triplets)
        :param return_set: flag indicating whether to return the triplets as a set (and filter out potential duplicates)
        :return: a list or a set of the triplets in the model's output string
        """
        collapsed_triplets_parts = cls._split_text(text, flatten=False)

        text_triplets = []
        for triplet_parts in collapsed_triplets_parts:
            try:
                current_sub = triplet_parts[0]
                rel_object_parts = triplet_parts[1:]
            except IndexError:
                log.info(
                    f"Textual sequence: ```{' '.join(triplet_parts)}``` cannot be parsed! "
                    f"Problematic triplet: {str(triplet_parts)}"
                )
                continue

            if verbose and len(rel_object_parts) % 2 != 0:
                # log.info(f"Textual sequence: ```{' '.join(triplet_parts)}``` does not follow the collapsed triplet
                # format!")
                log.info(
                    f"Textual sequence: ```{' '.join(triplet_parts)}``` does not follow the collapsed triplet format!"
                )
                rel_object_parts = rel_object_parts[:-1]

            it = iter(rel_object_parts)
            for rel in it:
                try:
                    obj = next(it)
                    text_triplets.append((current_sub, rel, obj))
                except StopIteration:
                    log.info(
                        f"Textual sequence: ```{' '.join(triplet_parts)}``` cannot be parsed! "
                        f"Problematic triplet: {str(triplet_parts)}"
                    )

        if not return_set:
            return text_triplets

        unique_text_triplets = set(text_triplets)

        if verbose and len(unique_text_triplets) != len(text_triplets):
            log.info(f"Textual sequence: ```{text}``` has duplicated triplets!")

        return unique_text_triplets


def remove_accents_from_str(input_str):
    nfkd_form = unicodedata.normalize("NFKD", input_str)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])
