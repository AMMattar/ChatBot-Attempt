from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
import string

tfd = TfidfTransformer()
decision = DecisionTreeClassifier()


def cleaner(x):
    return [a for a in (''.join([a for a in x if a not in string.punctuation])).lower().split()]


pipe = Pipeline([
    ('bow', CountVectorizer(analyzer=cleaner)),
    ('tfidf', tfd),
    ('classifier', decision)
])
