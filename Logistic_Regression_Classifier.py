import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

from utilities import visualize_classifier

X = np.array([[3.1,8.2],[4,7.7],[3.9,8],[5.5,4.7],[6,5],[5.6,5],[3.3,0.4],[3.9,0.9],[2.8,1],[0.5,3.4],[2,4],[0.8,4.7]])
y = np.array([0,0,0,1,1,1,2,2,2,3,3,3])
#Create LogisticRegression classifier
classifier = linear_model.LogisticRegression(solver='liblinear', C=100)
#Learning of classifier
classifier.fit(X,y)
#Visualize_classifier
visualize_classifier(classifier,X,y)
