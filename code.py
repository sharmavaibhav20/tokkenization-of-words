import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

# download necessary NLTK packages
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# example text
text = "This is sample example by Vaibhav."

# perform sentence tokenization
sentences = sent_tokenize(text)

# perform word tokenization and stopword removal
stop_words = set(stopwords.words('english'))
tokens = [word.lower() for sent in sentences for word in word_tokenize(sent) if word.lower() not in stop_words]

# perform stemming
ps = PorterStemmer()
stemmed_tokens = [ps.stem(token) for token in tokens]

# perform lemmatization
wnl = WordNetLemmatizer()
lemmatized_tokens = [wnl.lemmatize(token) for token in tokens]

print("Sentences:", sentences)
print("Tokens:", tokens)
print("Stemmed Tokens:", stemmed_tokens)
print("Lemmatized Tokens:", lemmatized_tokens)
