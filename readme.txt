1. Download p1_submission.zip from https://drive.google.com/file/d/1ewcjRk1OOb1SoQqMKibum_aZOANLdJfb/view?usp=sharing and extract all to the working path.

2. The environment file for anaconda is environment.yml

3. By default, the program can be executed by the command ./python SupervisedLearning.py
It will then 
	- loading data from dataset 1, online_shoppers_intention.csv and process 
	- generate graphs into ./graph
	- print performance statistic (if any) to console output
Same process will be repeated on dataset 2, census_income.csv 

4) To switch between different algorithms and dataset, main() in SupervisedLearning.py should be modified. A complete set of codes have been already written and commented by default. So uncommenting would allow execution on additional algorithms and datasets. For example, uncommenting the following code would run NeuralNetworks on dataset 1.

    data = loadData_1(encode_category = True)
    nn = NeuralNetworks(IDENTIFIER_1, FILENAME_1)
    nn.analyze(data)
	
5) By default, all algorithms will perform a full analysis. In case of a rerun on a particular chart or statistics, it can be done by commenting out functions from analyze() in [algorithm].py that are not required. For example, in analyze() of DecisionTree.py, commenting out first 3 lines will get the program print only the final decision tree and performance report.

	self.analyzeMaxDepthVsAccuracy(data)
	self.analyzeTrainSizeVsAccuracy(data)
	self.analyzeMaxDepthVsMaxTestingAccuracy(data)
	self.printDecisionTreeAndPerformance(data)
