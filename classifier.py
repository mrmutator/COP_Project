__author__ = 'rwechsler'
from sklearn import svm
from sklearn import naive_bayes
from corpora import get_swda_utterances_from_file
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split


X_tokens = []
Y_tags = []
for tag, tokens in get_swda_utterances_from_file("data/swda_file.txt"):
    X_tokens.append(" ".join(tokens))
    Y_tags.append(tag)

vectorizer = CountVectorizer(min_df=1)
X = vectorizer.fit_transform(X_tokens)

le = preprocessing.LabelEncoder()
le.fit(Y_tags)
Y = le.transform(Y_tags)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)


#model = svm.SVC()
model = naive_bayes.BernoulliNB()

model.fit(X_train, Y_train)
acc = model.score(X_train, Y_train)
print "Training accuracy", acc

acc = model.score(X_test, Y_test)
print "Test accuracy", acc



# predicted =  model.predict(X_test)
# true = le.inverse_transform(Y_test)
# for i, pred in enumerate(le.inverse_transform(predicted)):
#     print pred, true[i]
