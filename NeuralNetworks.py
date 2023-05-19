import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix
from util import splitData, getFullFilePath
from sklearn.model_selection import GridSearchCV
from time import time
from sklearn.metrics import accuracy_score
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

PRESET_HIDDEN_LAYERS = [(10, 10), (20, 20)]
PRESET_MAX_ITER = [300, 500]
PRESET_SOLVER = ['adam', 'sgd']

class NeuralNetworks:

    numeric_fields = []

    identifier = 0
    title = ''

    def __init__(self, id, title):
        self.identifier = id
        self.title = title

    def analyzeRunTime(self, data):

        X_train, X_test, y_train, y_test = splitData(data, 95, True)

        mlp = MLPClassifier(activation='relu', momentum=0.9,
            hidden_layer_sizes=PRESET_HIDDEN_LAYERS[self.identifier-1], learning_rate='constant',
            max_iter=PRESET_MAX_ITER[self.identifier-1], solver=PRESET_SOLVER[self.identifier-1])

        time_before_training = time()
        mlp.fit(X_train, y_train)
        time_after_training = time()
        predictions = mlp.predict(X_test)
        time_after_predict = time()

        print('NN %d training time: %f seconds.' % (self.identifier-1, time_after_training - time_before_training))
        print('NN %d lookup time: %f seconds.' % (self.identifier-1, time_after_predict - time_after_training))

        print(confusion_matrix(y_test, predictions))
        print(classification_report(y_test, predictions))


    def analyzeTrainSize(self, data):

        accuracy_train = []
        accuracy_test = []

        mlp = MLPClassifier(activation='relu', momentum=0.9,
            hidden_layer_sizes=PRESET_HIDDEN_LAYERS[self.identifier-1], learning_rate='constant',
            max_iter=PRESET_MAX_ITER[self.identifier-1], solver=PRESET_SOLVER[self.identifier-1],random_state=100)

        train_range = range(50, 100, 5)
        for train_size in train_range:

            X_train, X_test, y_train, y_test = splitData(data, train_size, True)
            mlp.fit(X_train, y_train)

            predict_train = mlp.predict(X_train)
            predict_test = mlp.predict(X_test)

            accuracy_train.append(accuracy_score(y_train, predict_train))
            accuracy_test.append(accuracy_score(y_test, predict_test))

        fig, ax = plt.subplots()

        ax.plot(train_range, accuracy_train, color="red", label="Training Accuracy")
        ax.plot(train_range, accuracy_test, color="green", label="Testing Accuracy")

        plt.title('Neural Networks: Training Size vs Accuracy on {title}'.format(title = self.title) )
        plt.xlabel('Training Size (%)')
        plt.ylabel("Accuracy")
        plt.legend()

        # stating max point
        ymax = max(accuracy_test)
        xpos = accuracy_test.index(ymax)
        xmax = train_range[xpos]
        ax.annotate('({xmax}, {ymax})'.format(xmax=xmax, ymax=round(ymax,4)), xy=(xmax, ymax), ha='center', va='bottom', color='green')

        filename = 'NN-{id}-TrainSize-vs-Accuracy.png'
        plt.savefig(getFullFilePath(filename.format(id = self.identifier)), bbox_inches='tight')
        plt.close()

    def analyzeBestParameter(self, data):

        X_train, X_test, y_train, y_test = splitData(data, 90, True)
        mlp = MLPClassifier(activation='relu', momentum=0.9,random_state=100)

        parameter_space = {
            'solver': ['adam', 'sgd'],
            'hidden_layer_sizes': [(5, 5), (10, 10), (15, 15), (20,20), (25,25), (5, 5, 5), (10, 10, 10), (15, 15, 15), (20,20,20), (10, 10, 10, 10)],
            'learning_rate': ['constant','adaptive'],
            'max_iter': [10, 100, 300, 500, 800],
        }

        clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=3)

        clf.fit(X_train, y_train)

        print('Best parameters found:\n', clf.best_params_)

        predictions = clf.predict(X_test)
        print(confusion_matrix(y_test, predictions))
        print(classification_report(y_test, predictions))


    def analyze(self, data):

        self.analyzeBestParameter(data)
        self.analyzeTrainSize(data)
        self.analyzeRunTime(data)





