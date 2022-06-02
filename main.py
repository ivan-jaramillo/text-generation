"""Text Generation

Generate text using a character-based recurrent neural network.
"""

import os
import time

import numpy as np
import tensorflow as tf


def split_input_target(sequence):
    input_text = sequence[:-1]
    target_text = sequence[1:]
    return input_text, target_text


def main():
    # Links to all of the datasets we'll use is here:
    # https://www.gutenberg.org/cache/epub/36/pg36.txt
    # https://www.gutenberg.org/files/84/84-0.txt
    # https://www.gutenberg.org/files/1342/1342-0.txt
    # https://www.gutenberg.org/files/1232/1232-0.txt
    # https://www.gutenberg.org/files/2554/2554-0.txt
    # https://www.gutenberg.org/cache/epub/1727/pg1727.txt
    # https://www.gutenberg.org/cache/epub/996/pg996.txt
    # https://www.gutenberg.org/files/3207/3207-0.txt
    # https://www.gutenberg.org/files/1998/1998-0.txt
    # https://www.gutenberg.org/cache/epub/64317/pg64317.txt
    
    # Hide TensorFlow message about it taking advantage of the CPU.
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    
    # Download one dataset.
    path_to_file = tf.keras.utils.get_file('war_of_the_worlds.txt', 'https://www.gutenberg.org/cache/epub/36/pg36.txt')
    text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
    print(f'The War of the Worlds has {len(text)} characters.')

    # Get the unique characters in the dataset.
    vocab = sorted(set(text))
    print(f'There are {len(vocab)} unique characters in The War of the Worlds.')

    # Vectorize the dataset.
    tokens = text.split()
    chars = tf.strings.unicode_split(tokens, input_encoding='UTF-8')
    ids_from_chars = tf.keras.layers.StringLookup(vocabulary=list(vocab), mask_token=None)

    # Obtain the vector of IDs from characters.
    ids = ids_from_chars(chars)
    print(ids)

    def text_from_ids(ids):
        return tf.strings.reduce_join(chars_from_ids(ids), axis=-1)

    # Recover characters from the vector of IDs.
    chars_from_ids = tf.keras.layers.StringLookup(vocabulary=ids_from_chars.get_vocabulary(), invert=True, mask_token=None)
    print(chars_from_ids)

    # Convert text vector into a stream of character indices.
    all_ids = ids_from_chars(tf.strings.unicode_split(text, 'UTF-8'))
    ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)
    seq_length = 100
    sequences = ids_dataset.batch(seq_length+1, drop_remainder=True)


if __name__ == '__main__':
    main()
