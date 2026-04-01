"""
Module for spelling feature engineering.

This module provides utilities to load a dictionary of valid words and calculate
the spelling error rate in a given text based on that dictionary.
"""

import re

def load_dict(path):
    """
    Loads a word list from a text file into a set for efficient lookup.

    Args:
        path (str): Path to the dictionary file (one word per line).

    Returns:
        set: A set of lowercase words from the dictionary file. Returns an
            empty set if the file is not found.
    """
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return set(word.strip().lower() for word in f)
    except FileNotFoundError:
        print(f"File {path} not found")
        return set()

def check_spelling(text, dict_set):
    """
    Calculates the ratio of misspelled words in a text.

    The text is normalized by converting to lowercase and removing non-alphabetic
    characters before splitting into individual words.

    Args:
        text (str): The input text to evaluate.
        dict_set (set): A set of valid lowercase words to check against.

    Returns:
        float: The error rate, calculated as (number of misspelled words) / (total words).
            Returns 0.0 if the dictionary is empty or if no words are found in the text.
    """
    if not dict_set:
        return 0.0
    
    cleaned_text = re.sub(r'[^a-z\s]', '', text.lower())
    words = cleaned_text.split()

    if not words:
        return 0.0
    
    error_count = sum(1 for word in words if word not in dict_set)
    return error_count / len(words)
