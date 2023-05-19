# <a name="_169nm0qpy49h"></a>Supervised Learning 

## <a name="_x932szno6yil"></a>Dataset
The datasets chosen for the supervised learning experiments are **Online Shoppers Intention** and **Census Income Data**. Data is taken from Machine Learning Repository [1], manually transformed and standardised to meet the experiment setting. 

As a sole trader in online business, **Online Shoppers Intention** is particularly interesting as it shows potential buyers behavior before their purchase. Features like **the time spent on different pages** and other Google Analytics data would possibly help building customer friendly content; others statistics such as **visiting month, operating systems**, and **region** may help to plan how marketing and development resources should be allocated.

**Census Income Data** shows a list of attributes factoring whether the income is greater than 50K annually. The dataset gives us a broader insight that it could be more than one thought that could contribute to income range, features listed are comprehensive including workclass, education, occupation, race, sex, working hours, etc. The data may help showing how a greater income can be achieved; or could be used as a reference for one’s credit assessment. 

In addition to that the datasets may broaden the knowledge on the relevant fields, its standard data structure featuring both continuous and categorical values, duplicate fields, and null values would be beneficial to my experience in data preparation for machine learning.
## <a name="_pwzdh715773s"></a>Decision Trees
The first algorithm implemented is **Decision Trees**, with python using machine learning library scikit-learn. There are two main challenges in data preparation.

**Categorical data handling** - scikit-learn assumes input data is continuous. If the input is non-numeric or numeric but actually representing categorical values, the result will be inaccurate due to the fact that a categorical value of 1 does not represent the average of 0 and 2. To rectify the issue, the data should be preprocessed with **one-hot encoding** [2] using panda function [3].

**Null data handling** - Data could be null because it’s confidential or missing. [4] Missing value of the numeric field is replaced by median; and that of the categorical field is replaced by Nan (as a new categorical value). As the number of records with missing value are relatively minor so the influence should be unnoticeable.
### <a name="_ldrsgi3lr86r"></a>Implementation
A decision tree should be constructed and tuned properly to maximize the accuracy. To simplify the process, I am applying the default parameters of DecisionTreeClassifier. [5] To avoid overfitting, **pruning** by specifying a maximum depth and **cross-validation** are used in following steps.

- Find optimal maximum depth (max-depth) of the decision tree with a fixed training size.
- Then with the obtained optimal depth, training with different size (train-size) and cross-validate with the remaining data for testing.
- Finally, scanning through varying maximum depth for its best accuracy to confirm the result.


|![](graph/e3b8080b-2ac4-4c39-9ab5-0a6169f74175.001.png)|![](graph/e3b8080b-2ac4-4c39-9ab5-0a6169f74175.002.png)|![](graph/e3b8080b-2ac4-4c39-9ab5-0a6169f74175.003.png)|
| :- | :- | :- |
|Figure 1.1.1|Figure 1.1.2|Figure 1.1.3|

The figures above illustrate the steps described on the dataset Online Shoppers Intention. Figure 1.1.1 shows overfitting occurs when maximum depth is greater approximately at 5, where the trained model fits better on training data but shows lower accuracy on testing data. By setting maximum depth to 5, figure 1.1.2 shows the learning curve, how the training size (in terms of percentage of the original dataset) may affect the accuracy by cross-validation. It shows that the greater volume of training data (beyond 80%) generally results in a better accuracy. To further verify the result, figure 1.1.3 shows how the best accuracy (of different training size) varys with different maximum depth. The experiment has proved the observation, giving the best accuracy at max-depth = 4 and train-size = 99%. 

In the second experiment, the same analysis has been done on Census Income Data, producing the following charts. Similarly figure 1.2.1 shows overfitting occurs beyond a maximum depth, approximately at 7. However, there is no obvious trend found at accuracy on varying training size in figure 1.2.2. The final chart figure 1.2.3 shows the best result is found at max-depth = 13 when the train-size = 98%.

Table 1 further compares the result on Online Shoppers Intention and Census Income Data.


|![](graph/e3b8080b-2ac4-4c39-9ab5-0a6169f74175.004.png)|![](graph/e3b8080b-2ac4-4c39-9ab5-0a6169f74175.005.png)|![](graph/e3b8080b-2ac4-4c39-9ab5-0a6169f74175.006.png)|
| :- | :- | :- |
|Figure 1.2.1|Figure 1.2.2|Figure 1.2.3|



||max-depth|train-size|accuracy|
| :- | :- | :- | :- |
|Online Shoppers Intention|4|99%|0\.91|
|Census Income Data|13|98%|0\.87|

Table 1 - best accuracy comparison

In general it is found that an average max-depth from 5 to 10 is able to generate a fairly acceptable accurate result, even in the dataset with up to 17 features. Apparently we would just need some of the dominant features to conclude the result. Moreover, the different learning curves show that the higher data size does not necessarily result in better accuracy.

Besides, a significant advantage of decision trees is the availability of comprehensive analysis of the consequences along each branch, as demonstrated in the figures below [6]. 


|![](graph/e3b8080b-2ac4-4c39-9ab5-0a6169f74175.007.png)|![](graph/e3b8080b-2ac4-4c39-9ab5-0a6169f74175.008.png)|
| :- | :- |
|Figure 1.3.1|Figure 1.3.2|


## <a name="_akgmxurl3y14"></a>Neural Networks

In this experiment, the Multi-Layer Perceptron Classifier model from the neural\_network library of scikit-learn is used as the estimator. The same categorical data handling as that of decision trees is needed as neural network processes only numeric values. In addition, the data has to be normalized to facilitate the convergence [7]. 
### <a name="_i1uuap6ov8ve"></a>Implementation
With scikit-learn, neural networks can be simply built with default parameters. However, it is a challenge to find out the best parameters combinations. While a general recommendation [8] can be used on some common datasets, the best setting of different datasets is always case by case. In my experiments, I will be looking at the following parameters: **solver** for weight optimization, **number of hidden layers and neurons**, **learning rate**, and **maximum iterations**.

I will be examining the default solver **adam** and **sgd** (stochastic gradient descent). For the number of layers and neurons, there is research [9][10][11] suggesting how it should be related to input size, and how the number of layers should be counted, so a number from 2 to 4 layers and from 5 to 25 neurons will be tested. Also I would like to verify if adapter learning rate could help avoiding suboptimal solution [12], and what number of iteration from 10 to 1000 would facilitate convergence without overfitting or suboptimal result by hitting local minima for example.

The testing will cover a broad range of non-linear parameter values. The hyper-parameter optimization tool GridSearchCV [13] is used to find the best parameters by looking for minimal error with cross validation. 

Due to the randomness of neural networks I expect indeterministic results in a sequence of executions on different dataset, as shown in Table 2.1 and Table 2.2, with accuracy calculated from confusion matrix [14] .


|**Run**|**hidden\_layer\_sizes**|**learning\_rate**|**max\_iter**|**solver**|**accuracy**|
| :- | :- | :- | :- | :- | :- |
|1|(5, 5, 5)|constant|300|adam|0\.88|
|2|(15,15)|constant|100|adam|0\.88|
|3|(15,15)|constant|100|adam|0\.89|

Table 2.1: Best parameters of neural networks on online\_shoppers\_intention.cs



|**Run**|**hidden\_layer\_sizes**|**learning\_rate**|**max\_iter**|**solver**|**accuracy**|
| :- | :- | :- | :- | :- | :- |
|1|(20, 20, 20)|constant|500|sgd|0\.84|
|2|(20, 20)|constant|700|sgd|0\.84|
|3|(25, 25)|adaptive|500|sgd|0\.84|

Table 2.2: Best parameters of neural networks on census\_income.cs

Later I realized the randomness can be avoided by setting **random state** for consistent outcome, as a result the consistent output is marked, as in table 2.3 below.


||**hidden\_layer\_sizes**|**learning\_rate**|**max\_iter**|**solver**|**accuracy**|
| :- | :- | :- | :- | :- | :- |
|**Online Shoppers Intention**|(20, 20)|constant|300|sgd|0\.89|
|**Census Income Data**|(25, 25)|constant|300|sgd|0\.84|

Table 2.3: Best parameters of neural networks with fixed random state

The results show that it is sufficient to take two **hidden layers**, impying they are problems more than just linear. **Number of neurons** used are less than the actual number of features, maybe some features do not contribute as much. **Learning rates** are varying due to randomness, and different **solvers** are adopted. By comparing the need of **the number of neurons and maximum iteration** to obtain the best accuracy, the dataset census\_income apparently has higher complexity.

Next I would like to further check if different train sizes would affect the accuracy through the learning curves. Figure 2.1 and 2.2 have shown maximum accuracy is found at different training sizes, but the difference is not significant in general.


|![](graph/e3b8080b-2ac4-4c39-9ab5-0a6169f74175.009.png)|![](graph/e3b8080b-2ac4-4c39-9ab5-0a6169f74175.010.png)|
| :- | :- |
|Figure 2.1|Figure 2.2|
## <a name="_ze1zrbz34k1z"></a>k-Nearest Neighbors
k-Nearest Neighbors (k-NN) is an instance based learning, also called lazy learning as computation is not required at training but done only at query. It relies on distance metric for classification, the performance is not known until the result is generated. 
### <a name="_r2hu9kljzudh"></a>Implementation 
In this experiment the value of k will be analyzed, for identifying the value for the best accuracy. Since we know already putting k equal to 1 would likely lead to overfitting, experiment is done by scanning on k from 2 to 50, with a step at 3 to identify an approximate value of k for the best accuracy. 

|![](graph/e3b8080b-2ac4-4c39-9ab5-0a6169f74175.011.png)|![](graph/e3b8080b-2ac4-4c39-9ab5-0a6169f74175.012.png)|
| :- | :- |
|Figure 3.1.1|Figure 3.1.2|

Figure 3.1.1 shows that a k-value of 8 would be able to achieve the best accuracy for Online Shoppers Intention. Figure 3.1.2 shows the best accuracy is found at k=23 for Census Income, starting from k=18 the accuracy looks acceptable though. Both results have implied a proper chosen k giving the well generalized classification, higher k value may not improve further but could be worse due to over generalization. Furthermore, the requirement of choosing different k in the datasets may depend on the noise, data density, and computation cost. 

Next I would examine how k-value is related with run time. Figure 3.2.1 and figure 3.2.2 both show increasing query time as growing k-value, as a result of accessing more neighbours for an average result.


|![](graph/e3b8080b-2ac4-4c39-9ab5-0a6169f74175.013.png)|![](graph/e3b8080b-2ac4-4c39-9ab5-0a6169f74175.014.png)|
| :- | :- |
|Figure 3.2.1|Figure 3.2.2|

With the acceptable result at k=8 and k=15, it is also interesting to see how their performance differs. In my opinion, the higher training time required on Census Income is due to:

1) More categorical fields resulting in more fields in total;
1) 3 times larger data sample volume

Significant time has been spent on saving data into the classifier structure.


||Online Shoppers Intention, k=8|Census Income, k=15|
| :- | :- | :- |
|Training time|0\.133811|1\.321519|
|Query time|0\.166447|0\.536672|
|Accuracy|0\.8|0\.78|

Table 3 - k-NN performance

As expected, the query time is much higher on Census Income because of higher k-value and more features are accessed when calculating distance metric. Generally speaking the accuracy is acceptable (higher than 0.5) but lower than the previous two algorithms, detailed comparison will be shown in the later section. One of the possible reasons is that the features may have different weighting, therefore normal (default) distance function may not work well. More research in distance function may generate higher accuracy.
## <a name="_mvetkmpa9ebh"></a>Boosting
From lecture we know that boosting ensemble meta-algorithms for reducing bias, and also variance in supervised learning, is used to convert weak learners to a strong learner. The experiment here is to examine boosting performance with its default parameter that the Decision Tree previously used is the base estimator and 50 estimators is adopted. The best maximum depth and training size found in the first experiment would be taken to contrast the performance of boosting under the same decision tree setting.


||**Decision Tree**|**Boosting**|
| :- | :- | :- |
||**Training time**|**Accuracy**|**Training time**|**Accuracy**|
|**Online Shoppers Intention**|0\.040975|0\.92|4\.302306|0\.90|
|**Census Income Data**|0\.228857|0\.84|3\.503842|0\.88|

Table 4.1 - best accuracy comparison between DT and Boosting

Table 4.1 shows the result of boosting with initial default parameters is not satisfactory on the first dataset. It is giving lower accuracy with much more time spent on training. Possible reasons are:

1) Overfitting: While the best accuracy is already found from standalone Decision Trees at the certain max-depth, boosting on the best performed model has made the model overfitting.
1) There are noises in the data. Incorrectly labelled in the train data may have built a worse model with boosting.
### <a name="_c3oo1hqj19al"></a>Experiment on Number of Estimators 

Then I wonder if the number of estimators would affect the outcomes. The graphs below shows 

how the number of estimators affect accuracy, training time and query time. 


|![](graph/e3b8080b-2ac4-4c39-9ab5-0a6169f74175.015.png)|![](graph/e3b8080b-2ac4-4c39-9ab5-0a6169f74175.016.png)|![](graph/e3b8080b-2ac4-4c39-9ab5-0a6169f74175.017.png)|
| :- | :- | :- |
|Figure 4.1.1|Figure 4.1.2|Figure 4.1.3|


|![](graph/e3b8080b-2ac4-4c39-9ab5-0a6169f74175.018.png)|![](graph/e3b8080b-2ac4-4c39-9ab5-0a6169f74175.019.png)|![](graph/e3b8080b-2ac4-4c39-9ab5-0a6169f74175.020.png)|
| :- | :- | :- |
|Figure 4.2.1|Figure 4.2.2|Figure 4.2.3|

The training and query time increase proportionally as expected. Regarding accuracy there are fluctuations with increasing number of estimators, but it looks like an upward trend on the second dataset.

If more estimators would help, I would be interested to see how a very high number of estimators at 10000 would perform, shown below in Table 4.2. However the accuracy has declined and much more training time were spent due to increased estimators. Again it looks like a result of overfitting and pink noise.


||**Training time**|**Query Time**|**Accuracy**|
| :- | :- | :- | :- |
|**Online Shoppers Intention**|327\.661869|3\.755384|0\.89|
|**Census Income Data**|1256\.448569|4\.476779|0\.82|

Table 4.2 - Performance of Boosting with n\_estimators = 10000
### <a name="_mg81i76o93m"></a>Experiment on Maximum Depth
If overfitting is an issue, the model has to be better generalized with pruning by setting different maximum depth. The charts below show how accuracy varies with maximum depths. With a proper setting of max-depth, the accuracy has improved to very close to standalone Decision Trees on both datasets; but with time sacrificed on repeated estimators, boosting anyway is not advantageous over Decision Trees.

As a result there is no way to further improve the current boosting result, the standalone decision tree has already generated a rather low-bias and low-variance model giving high accuracy. Also, it could be the pink noise that limits the overall accuracy improvement.


|![](graph/e3b8080b-2ac4-4c39-9ab5-0a6169f74175.021.png)|![](graph/e3b8080b-2ac4-4c39-9ab5-0a6169f74175.022.png)|
| :- | :- |
|Figure 4.3.1|Figure 4.3.2|
## <a name="_d8gw3on6743j"></a>Support Vector Machines
Different from the previous algorithms, I personally find the concept of Support Vector Machine (SVM) abstract and hard to understand from the way how the data is classified and queried. From research [15] I found some starting points such as the parameters **kernels**, their **coefficient (gamma)** if any, and the **regularization parameter (C)**.

Using **GridSearchCV**, I have searched through kernels **linear**, **rbf**, and **sigmoid**, with gamma either **scale** or **auto**, among C values **0.01, 0.1, 1, 10,** and **100.**

The first finding is that the data has to be normalized because svm works only on data in standard range, or the process will never end. The best parameter set for Online Shoppers Intention is **{'C': 10, 'gamma': 'scale', 'kernel': 'rbf'}**. With C=10, a lower regularization strength to ensure the model fits better the normalized training data. Gaussian RBF indicates the classification of data points is better measured from fixed point and gamma = scale (which is 1 / (n\_features \* X.var())) shows the variance of training data should be considered.

The best parameter set for Census Income Data is **{'C': 1, 'gamma': 'scale', 'kernel': 'linear'}**. This concludes that the data points can be linearly separated while gamma can be ignored. And no regularization is further required for the best result.

Secondly I will be examining how the value of regularization parameter C would affect the accuracy. The figures below show the experiment result, the declining accuracy is possibly due to over-generalization or overfitting with the improper regularization setting.


|![](graph/e3b8080b-2ac4-4c39-9ab5-0a6169f74175.023.png)|![](graph/e3b8080b-2ac4-4c39-9ab5-0a6169f74175.024.png)|
| :- | :- |
|Figure 5.1|Figure 5.2|

Finally there is the experiment to return the best accuracy and run time with the best parameters setting found in the previous experiments. Results will be shown in the next section and compared with other algorithms.
## <a name="_39bqot9xy9vg"></a>Comparison
### <a name="_o0i9ode0zzy6"></a>Run time
From Table 6.1 and 6.2, they show that both Neural Networks and Support Vector Machines require the highest training time due to their iterative nature, especially SVM’s training time grows exponentially with respect to the data size. NN spends less time on query though, once the weightings and bias are identified in the training phase. Boosting also requires relatively higher training time depending on the number of estimators. k-Nearest-Neighbour theoretically should not take too much time on training, but in reality when saving all instances into structure still consumes much IO time, compared with the time spent on Decision Trees training with pruning and limited maximum depth. In conclusion, DT gives the best performance in terms of time spent on training and query. It would be the best option for both the frequent need of model update and prediction query.


|**Time Spent**|**DT**|**NN**|**k-NN (k=8)**|**Boosting (max-depth:14)**|**SVM**|
| :- | :- | :- | :- | :- | :- |
|**Training**|0\.040975|17\.116292|0\.133811 |4\.271883|3\.379918|
|**Query**|0\.006996|0\.000999|0\.166447|0\.027980|0\.143910|
|**Iterations**|-|300|-|50|-|

Table 6.1: Run time of different algorithm on online\_shoppers\_intention.csv


|**Time Spent**|**DT**|**NN**|**k-NN (k=15)**|**Boosting (max-depth:2)**|**SVM**|
| :- | :- | :- | :- | :- | :- |
|**Training**|0\.228857|43\.708732|1\.321519|3\.357933|70\.038847|
|**Query**|0\.009996 |0\.001999|0\.536672|0\.034981|2\.102710|
|**Iterations**|-|500|-|50|-|

Table 6.2: Run time of different algorithm on census\_income.csv
### <a name="_n4cisf5bv4u9"></a>Performance Statistics
Table 7.1 and 7.2 show the performance statistics of different algorithms. Note the result includes accuracy, precision, recall, and f1-score [16]. Some classification results may cost higher than others sometimes, for example, a false negative is more expensive as a result of medical analysis. So it depends on the use case to determine which classification result should be taken. In our experiments, the Online Shoppers Intention and Census Income Data could be often used for marketing, I suppose the accuracy can be taken for comparison generally.


||**DT**|**NN**|**k-NN**|**Boosting**|**SVM**|
| :- | :- | :- | :- | :- | :- |
|**Accuracy**|0\.91|0\.89|0\.85|0\.90|0\.89|
|**Precision** |0\.91|0\.88|0\.83|0\.89|0\.88|
|**Recall**|0\.91|0\.89|0\.85|0\.90|0\.89|
|**f1-score**|0\.90|0\.88|0\.81|0\.88|0\.88|

Table 7.1 : performance statistics on online\_shoppers\_intention.csv



||**DT**|**NN**|**k-NN**|**Boosting**|**SVM**|
| :- | :- | :- | :- | :- | :- |
|**Accuracy**|0\.84|0\.85|0\.78|0\.88|0\.85|
|**Precision**|0\.83|0\.85|0\.77|0\.88|0\.84|
|**Recall**|0\.84|0\.85|0\.78|0\.88|0\.85|
|**f1-score**|0\.83|0\.85|0\.74|0\.88|0\.84|

Table 7.2 : performance statistics on census\_income.csv
## <a name="_h0uuhf3693a"></a>Conclusion
In conclusion, for the experiment of Online Shoppers Intention, **Decision Trees** is the best algorithm in terms of run time and accuracy. For Census Income Data, **Boosting** has the highest accuracy at 0.88 and training time better than NN and SVM. However, if the query time is the most crucial, then **Decision Trees** should be picked with its lowest query time yet acceptable accuracy at 0.84.

In general as explained in the previous sections, the choice of algorithms always depends on the use case, as such training time, query time, and all classification statistics would weigh differently influencing the choice of algorithms.

In our experiments, the number of features have been significantly increased with the use of one-hot encoding on categorical features, which may be the cause that instance-based classification has the worse performance. Moreover, there is an unusual case on the first dataset that Boosting cannot outrun Decision Trees, if DT has already generated a low-bias low-variance model. Generally in most cases we can always apply the same algorithms on the same classification problem in the short term, provided the data attributes does not change drastically; but a regular review on the choice of algorithms would be more beneficial in the long run.

## <a name="_tetf64tq0or6"></a>
## <a name="_zano7o41acvk"></a>Reference
1. *UCI Machine Learning Repository: Data Sets*, archive.ics.uci.edu/ml/datasets
1. Brownlee, Jason. “How to One Hot Encode Sequence Data in Python.” *Machine Learning Mastery*, 14 Aug. 2019, machinelearningmastery.com/how-to-one-hot-encode-sequence-data-in-python/
1. “Pandas.get\_dummies¶.” *Pandas.get\_dummies - Pandas 1.1.1 Documentation*, pandas.pydata.org/pandas-docs/stable/reference/api/pandas.get\_dummies.html
1. Angelov, Boyan. “Working with Missing Data in Machine Learning.” *Medium*, Towards Data Science, 13 Dec. 2017, towardsdatascience.com/working-with-missing-data-in-machine-learning-9c0a430df4ce
1. “How to Calculate Ideal Decision Tree Depth without Overfitting?” *Data Science Stack Exchange*, 1 June 1967, datascience.stackexchange.com/questions/26776/how-to-calculate-ideal-decision-tree-depth-without-overfitting
1. Galarnyk, Michael. “Visualizing Decision Trees with Python (Scikit-Learn, Graphviz, Matplotlib).” *Medium*, Towards Data Science, 2 Apr. 2020, towardsdatascience.com/visualizing-decision-trees-with-python-scikit-learn-graphviz-matplotlib-1c50b4aa68dc. 
1. Portilla, Jose. “A Beginner's Guide to Neural Networks in Python.” *Springboard Blog*, 2 Apr. 2019, www.springboard.com/blog/beginners-guide-neural-network-in-python-scikit-learn-0-18/
1. “MLPClassifier Parameter Setting.” *Stack Overflow*, 1 Jan. 1967, stackoverflow.com/questions/45769058/mlpclassifier-parameter-setting. 
1. Heaton, Jeff. “The Number of Hidden Layers.” *Heaton Research*, 20 July 2020, www.heatonresearch.com/2017/06/01/hidden-layers.html. 
1. Brownlee, Jason. “How to Configure the Number of Layers and Nodes in a Neural Network.” *Machine Learning Mastery*, 6 Aug. 2019, machinelearningmastery.com/how-to-configure-the-number-of-layers-and-nodes-in-a-neural-network/. 
1. Gad, Ahmed. “Beginners Ask ‘How Many Hidden Layers/Neurons to Use in Artificial Neural Networks?".” *Medium*, Towards Data Science, 27 June 2018, towardsdatascience.com/beginners-ask-how-many-hidden-layers-neurons-to-use-in-artificial-neural-networks-51466afa0d3e. 
1. Brownlee, Jason. “Understand the Impact of Learning Rate on Neural Network Performance.” *Machine Learning Mastery*, 25 Aug. 2020, machinelearningmastery.com/understand-the-dynamics-of-learning-rate-on-deep-learning-neural-networks/. 
1. “How to Adjust the Hyperparameters of MLP Classifier to Get More Perfect Performance.” *Data Science Stack Exchange*, 1 Dec. 1967, datascience.stackexchange.com/questions/36049/how-to-adjust-the-hyperparameters-of-mlp-classifier-to-get-more-perfect-performa. 
1. Konkiewicz, Magdalena. “Reading a Confusion Matrix.” *Medium*, Towards Data Science, 1 May 2020, towardsdatascience.com/reading-a-confusion-matrix-60c4dd232dd4. 
1. Sunil RayI am a Business Analytics and Intelligence professional with deep experience in the Indian Insurance industry. I have worked for various multi-national Insurance companies in last 7 years. “SVM: Support Vector Machine Algorithm in Machine Learning.” *Analytics Vidhya*, 15 Apr. 2020, www.analyticsvidhya.com/blog/2017/09/understaing-support-vector-machine-example-code/. 
1. Solutions, Exsilio, and \* Name. “Accuracy, Precision, Recall & F1 Score: Interpretation of Performance Measures.” *Exsilio Blog*, 11 Nov. 2016, blog.exsilio.com/all/accuracy-precision-recall-f1-score-interpretation-of-performance-measures/. 

## <a name="_x932szno6yil"></a>Running under Anaconda
1. The environment file for anaconda is environment.yml
1. By default, the program can be executed by the command `./python SupervisedLearning.py`
It will then 
	- loading data from dataset 1, online_shoppers_intention.csv and process 
	- generate graphs into ./graph
	- print performance statistic (if any) to console output
Same process will be repeated on dataset 2, census_income.csv 
1. To switch between different algorithms and dataset, `main()` in SupervisedLearning.py should be modified. A complete set of codes have been already written and commented by default. So uncommenting would allow execution on additional algorithms and datasets. For example, uncommenting the following code would run NeuralNetworks on dataset 1.

		data = loadData_1(encode_category = True)
		nn = NeuralNetworks(IDENTIFIER_1, FILENAME_1)
		nn.analyze(data)			
1. By default, all algorithms will perform a full analysis. In case of a rerun on a particular chart or statistics, it can be done by commenting out functions from `analyze()` in [algorithm].py that are not required. For example, in `analyze()` of DecisionTree.py, commenting out first 3 lines will get the program print only the final decision tree and performance report.

		self.analyzeMaxDepthVsAccuracy(data)
		self.analyzeTrainSizeVsAccuracy(data)
		self.analyzeMaxDepthVsMaxTestingAccuracy(data)
		self.printDecisionTreeAndPerformance(data)
