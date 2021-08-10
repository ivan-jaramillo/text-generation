import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing

import numpy as np
import os
import time


def main():
    text = open('data/the-great-gatsby.txt').read()
    text += open('data/the-war-of-the-worlds.txt').read()
    text += open('data/thus-spoke-zarathustra.txt').read()
    text += open('data/the-importance-of-being-earnest.txt').read()
    
    # Get unique characters from the dataset.
    vocab = sorted(set(text))

    # Vectorize the text.
    example_texts = ['abcdefg', 'xyz']
    chars = tf.strings.unicode_split(example_texts, input_encoding='UTF-8')
    print(chars)


if __name__ == '__main__':
    main()
