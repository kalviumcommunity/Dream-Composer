import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from tokenization.tokenizer import SimpleTokenizer

def test_tokenize_and_detokenize():
    t = SimpleTokenizer()
    text = "Hello world!"
    tokens = t.tokenize(text)
    assert tokens == ["Hello", "world!"]
    assert t.detokenize(tokens) == "Hello world!"
