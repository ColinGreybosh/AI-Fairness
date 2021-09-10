# AI-Fairness
This is the source code of ungoing AI fairness research

Install IBM AI Fairness 360 from: https://github.com/Trusted-AI/AIF360
Install current version of scikit-learn 
Running on Python 3.6.9

Command line argument: python fair_ml.py 

-d dataset including 'adult', 'compas', 'german', 'meps19', 'bank', 'grade', default is 'compas'

-c classifier including 'lr' (logistic regression), 'rf' (random forest), 'svm' (support vector machine), 'nn' (neural network), and 'nb' (Gaussian naive Bayes), default is 'lr'

-b amount of artificial bias introduced to the training set: [0,1.0], default is 0.


