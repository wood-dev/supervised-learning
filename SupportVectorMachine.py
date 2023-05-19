import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import accuracy_score
from util import splitData, getFullFilePath
import matplotlib.ticker as mticker
from sklearn import tree
from time import time
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import GridSearchCV

class SupportVectorMachine:

    identifier = 0
    title = ''

    PRESET_BEST_GAMMA = ['scale', 'scale']
    PRESET_BEST_KERNEL = ['rbf', 'linear']
    PRESET_BEST_C = [10, 1]

    def __init__(self, id, title):

        self.identifier = id
        self.title = title

    def analyzeRunTime(self, data):

        X_train, X_test, y_train, y_test = splitData(data, 90, True)

        clf = svm.SVC(gamma=self.PRESET_BEST_GAMMA[self.identifier-1], kernel=self.PRESET_BEST_KERNEL[self.identifier-1],C=self.PRESET_BEST_C[self.identifier-1],random_state=100)
        time_before_training = time()
        clf.fit(X_train, y_train)
        time_after_training = time()
        predictions = clf.predict(X_test)
        time_after_predict = time()

        print('SVM %d training time: %f seconds.' % (self.identifier-1, time_after_training - time_before_training))
        print('SVM %d lookup time: %f seconds.' % (self.identifier-1, time_after_predict - time_after_training))

        print(confusion_matrix(y_test, predictions))
        print(classification_report(y_test, predictions))

    def analyzeC(self, data):

        X_train, X_test, y_train, y_test = splitData(data, 90, True)

        accuracy_test = []

        c_range = [0.01, 0.1, 1, 10, 100]

        for c in c_range:

            clf = svm.SVC(gamma=self.PRESET_BEST_GAMMA[self.identifier-1], kernel=self.PRESET_BEST_KERNEL[self.identifier-1],C=c,random_state=100)
            clf = clf.fit(X_train, y_train)
            predict_test = clf.predict(X_test)
            accuracy_test.append(accuracy_score(y_test, predict_test))

        fig, ax = plt.subplots()

        plt.title('SVM : Regularization C vs Testing Accuracy on {title}'.format(title = self.title) )
        plt.xlabel('C')
        plt.ylabel("Testing Accuracy")
        ax.plot(c_range, accuracy_test, color="green", label="Testing Accuracy")
        ax.set_xscale('log')
        plt.legend()
        filename = 'SVM-{id}-1-C-vs-Accuracy.png'
        plt.savefig(getFullFilePath(filename.format(id = self.identifier)), bbox_inches='tight')
        plt.close()


    def analyzeBestParameter(self, data):

        print('SVM analyze best parameter: working on {filename}'.format(filename=self.title))

        X_train, X_test, y_train, y_test = splitData(data, 90, True)

        parameter_space = {
            'kernel': ['linear', 'rbf', 'sigmoid'],
            'gamma': ['scale', 'auto'],
            'C': [0.1, 1, 10],
        }

        svc = svm.SVC(random_state=100)
        clf = GridSearchCV(svc, parameter_space, n_jobs=-1, cv=3)
        clf.fit(X_train, y_train)

        print('Best parameters found:\n', clf.best_params_)
        predictions = clf.predict(X_test)
        print(confusion_matrix(y_test, predictions))
        print(classification_report(y_test, predictions))

    def analyze(self, data):

        self.analyzeBestParameter(data)
        self.analyzeC(data)
        self.analyzeRunTime(data)