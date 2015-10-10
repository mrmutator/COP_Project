__author__ = 'rwechsler'
from sklearn import svm
from sklearn import naive_bayes
from corpora import get_utterances_from_file
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def NB_classifier(X_train, Y_train, X_test, Y_test):
    model = naive_bayes.BernoulliNB()
    model.fit(X_train, Y_train)
    score = model.score(X_test, Y_test)

    return score


def SVM_classifier(X_train, Y_train, X_test, Y_test):
    SVM_model = svm.SVC(probability=False, kernel='rbf', C=1.0)
    SVM_model.fit(X_train,Y_train)
    score = SVM_model.score(X_test, Y_test)

    return score


def KNN_classifier(X_train, Y_train, X_test, Y_test):
    KNN_model = KNeighborsClassifier(n_neighbors=5)
    KNN_model.fit(X_train,Y_train)
    score = KNN_model.score(X_test,Y_test)

    return score


def get_swda_labeled_utterances():
    X_tokens = []
    Y_tags = []
    for tag, tokens in get_utterances_from_file("data/swda_file.txt"):
        X_tokens.append(" ".join(tokens))
        # remove id from tag
        tag = tag.split("/")[0]
        Y_tags.append(tag)

    return X_tokens, Y_tags


def get_BOW_from_utterances(X_tokens):
    # scikit learn BOW
    vectorizer = CountVectorizer(min_df=1)
    X = vectorizer.fit_transform(X_tokens) # returns a sparse matrix

    return X


def encode_tags(tags):
    # encode tags to ints
    le = preprocessing.LabelEncoder()
    le.fit(tags) # fit data
    Y = le.transform(tags) # return normalized tags

    return Y


def split_datasets(X, Y):
    # out of the box scikit method for splitting train/test. Get 10%
    return train_test_split(X, Y, test_size=0.1)


if __name__=='__main__':

    # get utterance tokens and tags from swda corpus
    X_tokens, Y_tags = get_swda_labeled_utterances()

    # get BOW for utterances
    X = get_BOW_from_utterances(X_tokens)

    # encode tags as ints
    Y = encode_tags(Y_tags)

    # get training and validation sets
    X_train, X_test, Y_train, Y_test = split_datasets(X,Y)

    # run classifiers
    NB_score = NB_classifier(X_train, Y_train, X_test, Y_test)
    # SVM_score = SVM_classifier(X_train, Y_train, X_test, Y_test)
    KNN_score = KNN_classifier(X_train, Y_train, X_test, Y_test)

    print 'NB test set score ', NB_score
    print 'KNN test set score: ', KNN_score

# predicted =  model.predict(X_test)
# true = le.inverse_transform(Y_test)
# for i, pred in enumerate(le.inverse_transform(predicted)):
#     print pred, true[i]
