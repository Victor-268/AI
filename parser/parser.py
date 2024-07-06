import nltk
import sys
from nltk.tokenize import word_tokenize
from nltk.tree import Tree

TERMINALS = """
Adj -> "country" | "dreadful" | "enigmatical" | "little" | "moist" | "red"
Adv -> "down" | "here" | "never"
Conj -> "and" | "until"
Det -> "a" | "an" | "his" | "my" | "the"
N -> "armchair" | "companion" | "day" | "door" | "hand" | "he" | "himself"
N -> "holmes" | "home" | "i" | "mess" | "paint" | "palm" | "pipe" | "she"
N -> "smile" | "thursday" | "walk" | "we" | "word"
P -> "at" | "before" | "in" | "of" | "on" | "to"
V -> "arrived" | "came" | "chuckled" | "had" | "lit" | "said" | "sat"
V -> "smiled" | "tell" | "were"
"""

NONTERMINALS = """
S -> NP VP | S Conj S
NP -> Det N | Det AP N | Det N PP | N | AP N | NP Conj NP | NP PP
AP -> Adj | Adj AP
PP -> P NP
VP -> V | V NP | V PP | V NP PP | Adv VP | VP Adv | VP Conj VP
"""

grammar = nltk.CFG.fromstring(NONTERMINALS + TERMINALS)
parser = nltk.ChartParser(grammar)


def main():

    # If filename specified, read sentence from file
    if len(sys.argv) == 2:
        with open(sys.argv[1]) as f:
            s = f.read()

    # Otherwise, get sentence as input
    else:
        s = input("Sentence: ")

    # Convert input into list of words
    s = preprocess(s)

    # Attempt to parse sentence
    try:
        trees = list(parser.parse(s))
    except ValueError as e:
        print(e)
        return
    if not trees:
        print("Could not parse sentence.")
        return

    # Print each tree with noun phrase chunks
    for tree in trees:
        tree.pretty_print()

        print("Noun Phrase Chunks")
        for np in np_chunk(tree):
            print(" ".join(np.flatten()))


def preprocess(sentence):
    """
    Convert `sentence` to a list of its words.
    Pre-process sentence by converting all characters to lowercase
    and removing any word that does not contain at least one alphabetic
    character.
    """
    words = word_tokenize(sentence.lower())

    filtered_words = []
    for word in words:
        contains_alpha = False
        for c in word:
            if c.isalpha():
                contains_alpha = True
                break
        if contains_alpha:
            filtered_words.append(word)
    return filtered_words


def np_chunk(tree):
    """
    Return a list of all noun phrase chunks in the sentence tree.
    A noun phrase chunk is defined as any subtree of the sentence
    whose label is "NP" that does not itself contain any other
    noun phrases as subtrees.
    """
    chunks = []

    # Define a helper function to recursively check for nested NPs
    def is_np_chunk(subtree):
        # Check if any child of this subtree is also an NP
        for child in subtree:
            if isinstance(child, Tree) and child.label() == "NP":
                return False
        return True

    # Iterate over all subtrees of the given tree
    for subtree in tree.subtrees():
        # Check if the subtree's label is "NP" and it is an NP chunk
        if subtree.label() == "NP" and is_np_chunk(subtree):
            chunks.append(subtree)

    return chunks


if __name__ == "__main__":
    main()
