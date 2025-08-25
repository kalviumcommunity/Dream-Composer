class SimpleTokenizer:
    def __init__(self):
        pass

    def tokenize(self, text: str):
        return text.split()

    def detokenize(self, tokens: list[str]):
        return " ".join(tokens)
