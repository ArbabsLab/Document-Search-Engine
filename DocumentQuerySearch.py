import string
import nltk
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Example documents
docs = [
    '''About us. We deliver Artificial Intelligence & Machine Learning
       solutions to solve business challenges.''',
    '''Contact information. Email [martin davtyan at filament dot ai]
       if you have any questions''',
    '''Filament Chat. A framework for building and maintaining a scalable
       chatbot capability''',
       'contact email to chat martin'
]

removePunctuation = str.maketrans({x: None for x in string.punctuation})
TOKENIZER = TreebankWordTokenizer()
STEMMER = PorterStemmer()


def StripWord(w):
    return [STEMMER.stem(token) for token in TOKENIZER.tokenize(w.translate(removePunctuation))]

vectorizer = TfidfVectorizer(tokenizer=StripWord, stop_words='english')
vectorizer.fit(docs)

query = 'contact email to chat martin'
queryVector = vectorizer.transform([query]).toarray()
docVector = vectorizer.transform(docs).toarray()
relevancy = cosine_similarity(queryVector, docVector)

ranking = (-relevancy).argsort(axis=None)

SearchResult = docs[ranking[0]]
print("Most Relevant Document #",ranking[0], ": ", SearchResult)
