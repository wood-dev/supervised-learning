import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,confusion_matrix
from util import splitData, getFullFilePath
from time import time
from sklearn.metrics import accuracy_score

PRESET_k = [8, 15]

class KNearestNeighbors:

    identifier = 0
    title = ''

    def __init__(self, id, title):
        self.identifier = id
        self.title = title

    def analyzeRunTime(self, data):

        X_train, X_test, y_train, y_test = splitData(data, 90)

        knc = KNeighborsClassifier(n_neighbors=PRESET_k[self.identifier-1])

        time_before_training = time()
        knc.fit(X_train, y_train)
        time_after_training = time()
        predictions = knc.predict(X_test)
        time_after_predict = time()

        print('kNN %d training time: %f seconds.' % (self.identifier-1, time_after_training - time_before_training))
        print('kNN %d lookup time: %f seconds.' % (self.identifier-1, time_after_predict - time_after_training))

        print(confusion_matrix(y_test, predictions))
        print(classification_report(y_test, predictions))


    def analyzeKVsRunTime(self, data):

        time_test = []

        k_range = range(2, 50, 3)
        for k in k_range:

            knc = KNeighborsClassifier(n_neighbors=k)

            X_train, X_test, y_train, y_test = splitData(data, 90)
            time_before_training = time()
            knc.fit(X_train, y_train)
            time_after_training = time()
            predict_test = knc.predict(X_test)
            time_after_predict = time()
            time_test.append(time_after_predict - time_after_training)

        fig, ax = plt.subplots()

        ax.plot(k_range, time_test, color="green", label="Query time on Testing")

        plt.title('k-Nearest-Neighbors : k-value vs Query Time on {title}'.format(title = self.title) )
        plt.xlabel('k-value')
        plt.ylabel("Query Time")
        plt.legend()

        filename = 'kNN-{id}-2-k-vs-RunTime.png'
        plt.savefig(getFullFilePath(filename.format(id = self.identifier)), bbox_inches='tight')
        plt.close()

    def analyzeBestK(self, data):

        X_train, X_test, y_train, y_test = splitData(data, 90)

        accuracy_train = []
        accuracy_test = []

        k_range = range(2, 50, 3)
        for k in k_range:

            knc = KNeighborsClassifier(n_neighbors=k)
            knc.fit(X_train, y_train)

            predict_train = knc.predict(X_train)
            predict_test = knc.predict(X_test)

            accuracy_train.append(accuracy_score(y_train, predict_train))
            accuracy_test.append(accuracy_score(y_test, predict_test))

        fig, ax = plt.subplots()

        ax.plot(k_range, accuracy_train, color="red", label="Training Accuracy")
        ax.plot(k_range, accuracy_test, color="green", label="Testing Accuracy")

        plt.title('k-Nearest-Neighbors : k-value vs Accuracy on {title}'.format(title = self.title) )
        plt.xlabel('k-value')
        plt.ylabel("Accuracy")
        plt.legend()

        # stating max point
        ymax = max(accuracy_test)
        xpos = accuracy_test.index(ymax)
        xmax = k_range[xpos]
        ax.annotate('({xmax}, {ymax})'.format(xmax=xmax, ymax=round(ymax,4)), xy=(xmax, ymax), ha='center', va='bottom', color='green')

        filename = 'kNN-{id}-1-k-vs-Accuracy.png'
        plt.savefig(getFullFilePath(filename.format(id = self.identifier)), bbox_inches='tight')
        plt.close()



    def analyze(self, data):

        self.analyzeBestK(data)
        self.analyzeKVsRunTime(data)
        self.analyzeRunTime(data)





