# Text Generation

A recurrent neural network implemented using TensorFlow for text generation at the character level.

## Description

Recurrent neural networks (RNNs) are artificial neural networks that can use previous outputs as inputs while having hidden states. RNNs are great at handling sequential data (such as text, speech, video, etc.) due to their internal memory.

Text generation is the use of natural language processing to generate text through software that is indistinguishable from that of text written by humans. It is the communication of ideas from data - something truly beautiful in and of itself. This can be an excellent tool for businesses, communication between medical professionals and patients, and our tools.

## How to Install

Use a virtual environment! It's good developer practice and allows you to have installed certain versions of dependencies for your Python projects. This leads to less headaches and saves time.

You can make one by doing the following in your terminal:

```shell
python3 -m venv env
```

You can activate the virtual environment by doing the following:

```shell
source env/bin/activate
```

To install dependencies this project requires, you can do the following:

```shell
pip install -r requirements.txt
```

You can deactivate the virtual environment by doing the following:

```shell
deactivate
```

## How to Use

You can run this program while in your virtual environment by doing the following:

```shell
python text-generation/main.py
```
