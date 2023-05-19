import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from util import splitData, getFullFilePath
import matplotlib.ticker as mticker
from sklearn import tree
from time import time
from sklearn.metrics import classification_report,confusion_matrix

class Boosting:

    PRESET_MAX_DEPTH = [4, 13]
    PRESET_BEST_TRAIN_SIZE = [99, 99]
    PRESET_BEST_MAX_DEPTH = [14, 2]

    identifier = 0
    title = ''

    def __init__(self, id, title):
        self.identifier = id
        self.title = title

    def analyzeStandardBoosting(self, data, estimator_number = 50, depth=-1):

        if depth == -1:
            depth = self.PRESET_MAX_DEPTH[self.identifier-1]

        X_train, X_test, y_train, y_test = splitData(data, self.PRESET_BEST_TRAIN_SIZE[self.identifier-1])
        clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=depth, random_state=100), n_estimators=estimator_number, random_state=100)

        time_before_training = time()
        clf = clf.fit(X_train, y_train)
        time_after_training = time()
        predict_test = clf.predict(X_test)
        time_after_predict = time()

        print('Boosting %d - n_estimator: %d, depth: %d' % (self.identifier-1, estimator_number, depth))
        print('Boosting %d training time: %f seconds.' % (self.identifier-1, time_after_training - time_before_training))
        print('Boosting %d lookup time: %f seconds.' % (self.identifier-1, time_after_predict - time_after_training))

        print(confusion_matrix(y_test, predict_test))
        print(classification_report(y_test, predict_test))

    def analyzeDepthVsAccuracy(self, data):

        X_train, X_test, y_train, y_test = splitData(data, self.PRESET_BEST_TRAIN_SIZE[self.identifier-1])

        accuracy_test = []
        train_time = []
        query_time = []

        depth_range = range(1, 20)

        for depth in depth_range:

            clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=depth, random_state=100), random_state=100)
            clf = clf.fit(X_train, y_train)
            predict_test = clf.predict(X_test)
            accuracy_test.append(accuracy_score(y_test, predict_test))


        fig, ax = plt.subplots()

        # stating max point
        ymax = max(accuracy_test)
        xpos = accuracy_test.index(ymax)
        xmax = depth_range[xpos]
        ax.annotate('({xmax}, {ymax})'.format(xmax=xmax, ymax=round(ymax,4)), xy=(xmax, ymax), ha='center', va='bottom', color='green')

        plt.title('Boosting : Maximum Depth vs Testing Accuracy on {title}'.format(title = self.title) )
        plt.xlabel('Maximum Depth')
        plt.ylabel("Testing Accuracy")
        ax.plot(depth_range, accuracy_test, color="green", label="Testing Accuracy")
        plt.legend()
        filename = 'BT-{id}-2-MaxDepth-vs-Accuracy.png'
        plt.savefig(getFullFilePath(filename.format(id = self.identifier)), bbox_inches='tight')
        plt.close()


    def analyzeEstimatorVsAccuracy(self, data):

        X_train, X_test, y_train, y_test = splitData(data, self.PRESET_BEST_TRAIN_SIZE[self.identifier-1])

        accuracy_test = []
        train_time = []
        query_time = []

        estimator_range = range(50, 500, 50)

        for estimator_no in estimator_range:

            clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=self.PRESET_MAX_DEPTH[self.identifier-1]),n_estimators=estimator_no)

            time_before_training = time()
            clf = clf.fit(X_train, y_train)
            time_after_training = time()
            predict_test = clf.predict(X_test)
            time_after_predict = time()

            accuracy_test.append(accuracy_score(y_test, predict_test))
            train_time.append(time_after_training - time_before_training)
            query_time.append(time_after_predict - time_after_training)


        plt.title('Boosting : Number of Estimators vs Testing Accuracy on {title}'.format(title = self.title) )
        plt.xlabel('Estimators Number')
        plt.ylabel("Testing Accuracy")
        plt.plot(estimator_range, accuracy_test, color="green", label="Testing Accuracy")
        plt.legend()
        filename = 'BT-{id}-1-Estimator-vs-Accuracy.png'
        plt.savefig(getFullFilePath(filename.format(id = self.identifier)), bbox_inches='tight')
        plt.close()

        plt.title('Boosting : Number of Estimators vs Training Time on {title}'.format(title = self.title) )
        plt.xlabel('Estimators Number')
        plt.ylabel("Training Time")
        plt.plot(estimator_range, train_time, color="green", label="Training Time (s)")
        plt.legend()
        filename = 'BT-{id}-1-Estimator-vs-TrainingTime.png'
        plt.savefig(getFullFilePath(filename.format(id = self.identifier)), bbox_inches='tight')
        plt.close()

        plt.title('Boosting : Number of Estimators vs Query Time on {title}'.format(title = self.title) )
        plt.xlabel('Estimators Number')
        plt.ylabel("Query Time")
        plt.plot(estimator_range, query_time, color="green", label="Query Time (s)")
        plt.legend()
        filename = 'BT-{id}-1-Estimator-vs-QueryTime.png'
        plt.savefig(getFullFilePath(filename.format(id = self.identifier)), bbox_inches='tight')
        plt.close()

    def analyze(self, data):

        self.analyzeStandardBoosting(data)
        self.analyzeEstimatorVsAccuracy(data)
        self.analyzeStandardBoosting(data, 10000)
        self.analyzeDepthVsAccuracy(data)
        self.analyzeStandardBoosting(data, 50, self.PRESET_BEST_MAX_DEPTH[self.identifier-1])