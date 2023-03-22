import nltk
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.tree import Tree

# download necessary NLTK packages
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# sample sentence
sentence = "This is sample by example by Vaibhav."

# perform word tokenization
words = word_tokenize(sentence)

# perform POS tagging
pos_tags = pos_tag(words)

# construct a tree structure for POS tagged output
root = Tree('ROOT', [])
for word, tag in pos_tags:
    subtree = Tree(tag, [word])
    root.append(subtree)

# display POS tagged output in a tree format
print(root.pretty_print())
