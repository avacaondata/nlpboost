import re
import tokenizers
import evaluate
import collections
from tqdm import tqdm
import numpy as np
from functools import partial
from .utils import match_questions_multiple_answers


def tokenize_classification(examples, tokenizer, dataset_config):
    """
    Tokenize classification datasets.

    Given a dataset, a tokenizer and a dataset configuration, returns
    the tokenized dataset.

    Parameters
    ----------
    examples: datasets.Dataset
        Samples from datasets.Dataset.
    tokenizer: tokenizers.Tokenizer
        Instance of hf's tokenizer.
    dataset_config: benchmarker.DatasetConfig
        Instance of a Dataset Config.

    Returns
    -------
    tokenized:
        Tokenized samples.
    """
    if dataset_config.is_2sents:
        tokenized = tokenizer(
            examples[dataset_config.sentence1_field],
            examples[dataset_config.sentence2_field],
            truncation=True,
            padding="longest",
            max_length=512,
        )
    else:
        tokenized = tokenizer(
            examples[dataset_config.text_field],
            truncation=True,
            padding="longest",
            max_length=512,
        )
    if not dataset_config.is_multilabel:
        tokenized["labels"] = examples[dataset_config.label_col]
    else:
        columns_not_text = list(
            sorted([col for col in examples if dataset_config.text_field not in col])
        )
        labels = [
            [float(examples[col][i]) for col in columns_not_text]
            for i in range(len(examples[dataset_config.text_field]))
        ]
        tokenized["labels"] = labels
    return tokenized


def tokenize_ner(examples, tokenizer, dataset_config):
    """
    Tokenize a dataset or dataset split.

    This function is intended to be used inside the map method for the Dataset.

    Parameters
    ----------
    examples: datasets.Dataset
        Samples from datasets.Dataset.
    tokenizer: tokenizers.Tokenizer
        Instance of hf's tokenizer.
    dataset_config: benchmarker.DatasetConfig
        Instance of a Dataset Config.

    Returns
    -------
    tokenized:
        Tokenized samples.
    """
    ignore_index = -100
    tokenized = tokenizer(
        examples[dataset_config.text_field],
        truncation=True,
        is_split_into_words=True,
        padding="longest",
        max_length=512,
    )

    labels = []
    for i, label in enumerate(examples[dataset_config.label_col]):
        word_ids = tokenized.word_ids(batch_index=i)
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so
            # they are automatically ignored in the loss function.
            if word_idx is None:
                label_ids.append(ignore_index)
            else:
                label_ids.append(label[word_idx])
        labels.append(label_ids)

    tokenized["labels"] = labels
    return tokenized


def tokenize_squad(examples, tokenizer, dataset_config=None, pad_on_right=True):
    """
    Tokenize samples of squad-like datasets, on batches.

    It differentiates between BPE tokenizers and others
    as there are errors in these ones if they are processed in the conventional way.

    Parameters
    ----------
    examples: datasets.Dataset
        Samples from datasets.Dataset.
    tokenizer: tokenizers.Tokenizer
        Instance of hf's tokenizer.
    pad_on_right: bool
        Whether or not to pad the samples on the right side. True for most models.

    Returns
    -------
    tokenized_examples:
        Tokenized samples.
    """
    tokenized_examples = tokenizer(
        examples["question" if pad_on_right else "context"],
        examples["context" if pad_on_right else "question"],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=512,
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    # The offset mappings will give us a map from token to character position in the original context. This will
    # help us compute the start_positions and end_positions.
    offset_mapping = tokenized_examples.pop("offset_mapping")
    # Let's label those examples!
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []
    for i, offsets in enumerate(offset_mapping):
        # We will label impossible answers with the index of the CLS token.
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]
        # If no answers are given, set the cls_index as answer.
        if len(answers["answer_start"]) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            # Start/end character index of the answer in the text.
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])
            # Start token index of the current span in the text.
            token_start_index = 0
            while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                token_start_index += 1

            # End token index of the current span in the text.
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                token_end_index -= 1

            # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
            if not (
                offsets[token_start_index][0] <= start_char
                and offsets[token_end_index][1] >= end_char
            ):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                # Note: we could go after the last offset if the answer is the last word (edge case).
                while (
                    token_start_index < len(offsets)
                    and offsets[token_start_index][0] <= start_char
                ):
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)
    return tokenized_examples


def tokenize_summarization(examples, tokenizer, dataset_config):
    """
    Tokenization function for summarization tasks.

    Parameters
    ----------
    examples: datasets.Dataset
        Samples from datasets.Dataset.
    tokenizer: tokenizers.Tokenizer
        Instance of hf's tokenizer.
    dataset_config: benchmarker.DatasetConfig
        Instance of a Dataset Config.

    Returns
    -------
    examples: datasets.Dataset
        Tokenized samples with all necessary fields.
    """
    model_inputs = tokenizer(
        examples[dataset_config.text_field],
        truncation=True,
        max_length=tokenizer.model_max_length,
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples[dataset_config.summary_field],
            max_length=dataset_config.max_length_summary,
            truncation=True,
        )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
