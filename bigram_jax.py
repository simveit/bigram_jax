import random
from typing import Any, List, NamedTuple, Tuple, Union

import jax
import numpy as np

### Data Loading, Encoding, Decoding

def load_data(path: str) -> Tuple[List[str], List[str]]:

    with open(path, 'r') as f:
        data = f.read()

    words = data.splitlines()
    words = [word.strip() for word in words] # Remove leading/trailing whitespace
    words = [word for word in words if word] # Remove empty strings

    vocab = sorted(list(set(''.join(words))))
    vocab = ['<eos>'] + vocab
    print(f"number of examples in dataset: {len(words)}")
    print(f"max word length: {max([len(word) for word in words])}")
    print(f"min word length: {min([len(word) for word in words])}")
    print(f"unique characters in dataset: {len(vocab)}")
    print("vocabulary:")
    print(' '.join(vocab))
    print('example for a word:')
    print(words[0])
    return words, vocab

def encode(word: str, vocab: List[str]) -> List[int]:
    """
    Encode a word, add <eos> at the beginning and the end of the word.
    """
    return [vocab.index('<eos>')] + [vocab.index(char) for char in word] + [vocab.index('<eos>')]

def decode(indices: List[int], vocab: List[str]) -> str:
    """
    Decode a list of indices to a word using the vocabulary.
    """
    return ''.join([vocab[index] for index in indices])

def get_dataset(encoded_words: List[List[int]]) -> Tuple[jax.Array, jax.Array]:
    """
    Convert a list of encoded words to a list of bigrams.
    """
    X = []
    y = []
    for word in encoded_words:
        for char1, char2 in zip(word[:-1], word[1:]):
            X.append(char1)
            y.append(char2)
    return jax.numpy.array(X), jax.numpy.array(y)

def get_train_val_test(encoded_words: List[List[int]]) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """
    Split the dataset into training, validation and test sets.
    """
    random.shuffle(encoded_words)
    train_words = encoded_words[:int(0.8*len(encoded_words))]
    val_words = encoded_words[int(0.8*len(encoded_words)):int(0.9*len(encoded_words))]
    test_words = encoded_words[int(0.9*len(encoded_words)):]
    X_train, y_train = get_dataset(train_words)
    X_val, y_val = get_dataset(val_words)
    X_test, y_test = get_dataset(test_words)
    return X_train, y_train, X_val, y_val, X_test, y_test

### Modelling

class Weights(NamedTuple):
    W: jax.Array

def init_weights(vocab_size: int) -> Weights:
    return Weights(W=jax.numpy.array(np.random.randn(vocab_size, vocab_size)))

def forward(weights: Weights, X: jax.Array, return_logits: bool = False) -> jax.Array:
    """
    1) index into the weights matrix W using the input indices
    2) apply the softmax function to obtain a probability distribution over the next character.
    """
    logits = weights.W[X]
    if return_logits:
        return logits
    exp_logits = jax.numpy.exp(logits)
    probs = exp_logits / jax.numpy.sum(exp_logits, axis=1, keepdims=True)
    return probs

def loss(weights: Weights, X: jax.Array, y: jax.Array) -> jax.Array:
    """
    1) get the probabilities for the next character
    2) index into the probabilities using the true next character
    3) take the negative log of the probability
    4) return the mean loss over all the examples
    """
    probs = forward(weights, X)
    return -jax.numpy.log(probs[jax.numpy.arange(len(y)), y]).mean()

def update(weights: Weights, X: jax.Array, y: jax.Array, learning_rate: float) -> Union[Weights, Any]:
    """
    1) get the probabilities for the next character
    2) compute the gradient of the loss with respect to the weights
    3) update the weights using the gradient
    """
    grads = jax.grad(loss)(weights, X, y)
    return jax.tree.map(lambda w, g: w - learning_rate * g, weights, grads)

@jax.jit
def train_step(weights: Weights, X: jax.Array, y: jax.Array, learning_rate: float) -> Tuple[Weights, Union[Any, jax.Array]]:
    """
    1) compute the loss
    2) compute the gradient of the loss with respect to the weights
    3) update the weights using the gradient
    4) return the updated weights and the loss
    """
    loss_value = loss(weights, X, y)
    weights = update(weights, X, y, learning_rate)
    return weights, loss_value

def train(weights: Weights, X_train: jax.Array, y_train: jax.Array, X_val: jax.Array, y_val: jax.Array, learning_rate: float, N_EPOCHS: int) -> Weights:
    """
    1) loop over the number of epochs
    2) for each epoch, loop over the training data and update the weights
    3) compute the validation loss
    4) print the loss and validation loss
    """
    for epoch in range(N_EPOCHS):
        weights, loss_value = train_step(weights, X_train, y_train, learning_rate)
        val_loss = loss(weights, X_val, y_val)
        if epoch % 10 == 0:
            print(f"epoch: {epoch}, loss: {loss_value}, val_loss: {val_loss}")
    return weights

def sample(weights: Weights, key: jax.Array, vocab: List[str]) -> str:
    """
    1) Start with <eos>
    2) Index into the weights matrix W for the current character
    3) Sample the next character from the distribution
    4) Append the sampled character to the sampled word
    5) Repeat steps 3-5 until <eos> is sampled
    6) Return the sampled word
    """
    sampled_word = ['<eos>']
    current_char = jax.numpy.array([vocab.index('<eos>')])
    while True:
        key, subkey = jax.random.split(key)
        logits = forward(weights, current_char, return_logits=True)[0]
        next_char = jax.random.categorical(subkey, logits)
        next_char_int = int(next_char)
        sampled_word.append(vocab[next_char_int])
        if next_char_int == vocab.index('<eos>'):
            break
        current_char = jax.numpy.array([next_char_int])
    return ''.join(sampled_word[1:-1])  # Remove start and end <eos> tokens

if __name__ == "__main__":
    words, vocab = load_data('names.txt')
    encoded_words = [encode(word, vocab) for word in words]
    print(f"Encoding from {words[0]} -> {encoded_words[0]}")
    X_train, y_train, X_val, y_val, X_test, y_test = get_train_val_test(encoded_words)
    print("Built train, validation and test sets")
    print(f"# training examples: {len(X_train)}")
    print(f"# validation examples: {len(X_val)}")
    print(f"# test examples: {len(X_test)}")
    weights = init_weights(len(vocab))
    trained_weights = train(weights, X_train, y_train, X_val, y_val, 50, 100)
    print("Sanity check: Compare words generated from trained_weights and untrained_weights")
    for i in range(10):
        key = jax.random.PRNGKey(i)
        print(f"word from untrained weights: {sample(weights, key, vocab)}")
        key, subkey = jax.random.split(key)
        print(f"word from trained weights: {sample(trained_weights, key, vocab)}")
        print("#"*30)
