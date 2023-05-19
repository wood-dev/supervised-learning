import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from util import splitData, getFullFilePath
import matplotlib.ticker as mticker
from sklearn import tree
from time import time
from sklearn.metrics import classification_report,confusion_matrix

class DecisionTree:

    PRESET_CRITERION = 'gini'
    PRESET_SPLITTER = 'best'
    PRESET_MAX_DEPTH = [5, 5]

    PRESET_BEST_MAX_DEPTH = [5, 9]
    PRESET_BEST_TRAIN_SIZE = [99, 99]

    identifier = 0
    title = ''

    def __init__(self):
        pass

    def __init__(self, id, title):
        self.identifier = id
        self.title = title

    def analyzeMaxDepthVsAccuracy(self, data):

        accuracy_train = []
        accuracy_test = []
        max_depth_range = range(1, 50)
        for max_depth in max_depth_range:

            classifier_dt = DecisionTreeClassifier(max_depth=max_depth, random_state=100)
            X_train, X_test, y_train, y_test = splitData(data, 90)
            classifier_dt = classifier_dt.fit(X_train, y_train)
            predict_train = classifier_dt.predict(X_train)
            accuracy_train.append(accuracy_score(y_train, predict_train))
            predict_test = classifier_dt.predict(X_test)
            accuracy_test.append(accuracy_score(y_test, predict_test))

        plt.title('Decision Tree : Max Depth vs Training Accuracy on {title}'.format(title = self.title) )
        plt.xlabel('Max Depth')
        plt.ylabel("Training Accuracy")
        plt.plot(max_depth_range, accuracy_train, color="red", label="Training Accuracy")
        plt.plot(max_depth_range, accuracy_test, color="green", label="Testing Accuracy")
        plt.legend()
        filename = 'DT-{id}-1-MaxDepth-vs-Accuracy.png'
        plt.savefig(getFullFilePath(filename.format(id = self.identifier)), bbox_inches='tight')
        plt.close()


    def analyzeTrainSizeVsAccuracy(self, data):

        accuracy_train = []
        accuracy_test = []
        classifier_dt = DecisionTreeClassifier(max_depth=self.PRESET_MAX_DEPTH[self.identifier-1],random_state=100)

        train_range = range(40, 100)
        for train_size in train_range:

            X_train, X_test, y_train, y_test = splitData(data, train_size)
            classifier_dt = classifier_dt.fit(X_train, y_train)

            predict_train = classifier_dt.predict(X_train)
            predict_test = classifier_dt.predict(X_test)

            accuracy_train.append(accuracy_score(y_train, predict_train))
            accuracy_test.append(accuracy_score(y_test, predict_test))

        fig, ax = plt.subplots()

        ax.plot(train_range, accuracy_train, color="red", label="Training Accuracy")
        ax.plot(train_range, accuracy_test, color="green", label="Testing Accuracy")

        plt.title('Decision Tree : Training Size vs Accuracy under max_depth = {max_depth} on {title}'.format(max_depth = self.PRESET_MAX_DEPTH[self.identifier-1], title = self.title) )
        plt.xlabel('Training Size (%)')
        plt.ylabel("Accuracy")
        plt.legend()

        # stating max point
        ymax = max(accuracy_test)
        xpos = accuracy_test.index(ymax)
        xmax = train_range[xpos]
        ax.annotate('({xmax}, {ymax})'.format(xmax=xmax, ymax=round(ymax,4)), xy=(xmax, ymax), ha='center', va='bottom', color='green')

        filename = 'DT-{id}-2-TrainSize-vs-Accuracy.png'
        plt.savefig(getFullFilePath(filename.format(id = self.identifier)), bbox_inches='tight')
        plt.close()

    def analyzeMaxDepthVsMaxTestingAccuracy(self, data):

        max_accuracy_test = []
        best_train_size_list = []

        max_depth_range = range(1, self.PRESET_MAX_DEPTH[self.identifier-1]+10)
        for max_depth in max_depth_range:

            accuracy_test = []
            classifier_dt = DecisionTreeClassifier(max_depth=max_depth,random_state=100)

            train_range = range(40, 100)
            for train_size in train_range:

                X_train, X_test, y_train, y_test =  splitData(data, train_size)
                classifier_dt = classifier_dt.fit(X_train, y_train)
                predict_test = classifier_dt.predict(X_test)
                accuracy_test.append(accuracy_score(y_test, predict_test))

            best_accuracy_test = max(accuracy_test)
            xpos = accuracy_test.index(best_accuracy_test)
            xmax = train_range[xpos]
            best_train_size_list.append(xmax)

            max_accuracy_test.append(best_accuracy_test)

        fig, ax = plt.subplots()
        plt.title('Decision Tree : Max Depth vs Training Accuracy on {title}'.format(title = self.title))
        plt.xlabel('Max Depth')
        plt.ylabel("Training Accuracy")
        ax.plot(max_depth_range, max_accuracy_test, color="green", label="Test Accuracy")
        plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
        plt.legend()

        best_accuracy = max(max_accuracy_test)
        xpos = max_accuracy_test.index(best_accuracy)
        best_max_depth = max_depth_range[xpos]
        best_train_size = best_train_size_list[xpos]

        ax.annotate('best-depth: {xmax}\nbest-accuracy: {ymax}\nbest-train-size: {train_size}'.format(xmax=best_max_depth, ymax=round(best_accuracy,4), train_size=best_train_size),
            xy=(best_max_depth, best_accuracy), ha='left', va='top', color='green')

        filename = 'DT-{id}-3-MaxDepth-vs-MaxTestingAccuracy.png'
        plt.savefig(getFullFilePath(filename.format(id = self.identifier)), bbox_inches='tight')
        plt.close()

    def printDecisionTreeAndPerformance(self, data):

        X_train, X_test, y_train, y_test = splitData(data, self.PRESET_BEST_TRAIN_SIZE[self.identifier-1])
        classifier_dt = DecisionTreeClassifier(max_depth=self.PRESET_BEST_MAX_DEPTH[self.identifier-1],random_state=100)

        time_before_training = time()
        classifier_dt = classifier_dt.fit(X_train, y_train)
        time_after_training = time()
        predict_test = classifier_dt.predict(X_test)
        time_after_predict = time()

        X = data.iloc[:,:-1]
        y = data.iloc[:, -1]
        fn = list(X.columns.values)
        cn = list(map(str, set(y)))

        fig, axes = plt.subplots(figsize=(12, 12), dpi=600)
        tree.plot_tree(classifier_dt, feature_names = fn, class_names=cn, filled = True, fontsize = 10, max_depth = 4)
        fig.savefig(getFullFilePath('DT-{id}-GeneratedTree.png'.format(id = self.identifier)))
        plt.close()

        print('DT %d training time: %f seconds.' % (self.identifier-1, time_after_training - time_before_training))
        print('DT %d lookup time: %f seconds.' % (self.identifier-1, time_after_predict - time_after_training))

        print(confusion_matrix(y_test, predict_test))
        print(classification_report(y_test, predict_test))

    def analyze(self, data):

        self.analyzeMaxDepthVsAccuracy(data)
        self.analyzeTrainSizeVsAccuracy(data)
        self.analyzeMaxDepthVsMaxTestingAccuracy(data)
        self.printDecisionTreeAndPerformance(data)

