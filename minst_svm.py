import gzip, pickle
windowsPath = "E:\MegaSync\MEGAsync\python\datamining\Minst\mnist.pkl.gz"
MacPath = "/Users/cyh/Documents/megaSync/python/datamining/mnist.pkl.gz"
with gzip.open(windowsPath,'rb') as ff:
   u = pickle._Unpickler(ff)
   u.encoding = 'latin1'
   datasets =(u.load())

# Library
from sklearn.preprocessing import scale
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

# Data split
X_train = datasets[0][0]
Y_train = datasets[0][1]
X_test = datasets[1][0]
Y_test = datasets[1][1]

# Matrix to 28 x 28
X_train_metrix = []
for i in datasets[0][0]:
    metric = np.reshape(i[:],(28,28))
    X_train_metrix.append(metric)

# Import GridSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn import svm
# Set the parameter candidates
parameter_candidates = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},{'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},]
# Create a classifier with the parameter candidates
clf = GridSearchCV(estimator=svm.SVC(), param_grid=parameter_candidates, n_jobs=-1)
# Train the classifier on training data
clf.fit(X_train, Y_train)
print('Best score for training data:', clf.best_score_)
print('Best `C`:',clf.best_estimator_.C)
print('Best kernel:',clf.best_estimator_.kernel)
print('Best `gamma`:',clf.best_estimator_.gamma)

# SVM
import datetime
start_time = datetime.datetime.now()
from sklearn import svm
svc_model = svm.SVC(gamma=0.001, C=100., kernel='rbf')
svc_model.fit(X_train, Y_train)
end_time = datetime.datetime.now()
predicted = svc_model.predict(X_test)
print("Spending time : %s " % (end_time-start_time))

# Scoring
from sklearn.metrics import confusion_matrix
table = confusion_matrix(predicted, Y_test)
score = 100*(np.diag(table).sum()/table.sum())
print("The score for SVM model is {:.1f}%".format(score))
# Plot

fig = plt.figure(figsize=(8, 3))
fig.suptitle('Cluster Center Images', fontsize=14, fontweight='bold')

# We got ten points for ten clusters, they are values. So we could plot and see what happened.
for i in range(10):
    ax = fig.add_subplot(2, 5, i + 1)
    ax.imshow(clf.cluster_centers_[i].reshape((28, 28)), cmap=plt.cm.binary)
    plt.axis('off')

# Isomap
from sklearn.manifold import Isomap
X_iso = Isomap(n_neighbors=10).fit_transform(X_train)

fig, ax = plt.subplots(1, 2, figsize=(8, 4))

fig.suptitle('Predicted Versus Training Labels(PCA)', fontsize=14, fontweight='bold')
fig.subplots_adjust(top=0.85)

ax[0].scatter(X_iso[:, 0], X_iso[:, 1], c=clusters)
ax[0].set_title('Predicted Training Labels')
ax[1].scatter(X_iso[:, 0], X_iso[:, 1], c=Y_train)
ax[1].set_title('Actual Training Labels')

plt.show()

# PCA
from sklearn.decomposition import PCA
randomized_pca = PCA(n_components=2)
reduce_data_pca = randomized_pca.fit_transform(X_train)

fig, ax = plt.subplots(1, 2, figsize=(8, 4))

fig.suptitle('Predicted Versus Training Labels(PCA)', fontsize=14, fontweight='bold')
fig.subplots_adjust(top=0.85)

ax[0].scatter(reduce_data_pca[:, 0], reduce_data_pca[:, 1], c=predicted)
ax[0].set_title('Predicted Training Labels')
ax[1].scatter(reduce_data_pca[:, 0], reduce_data_pca[:, 1], c=Y_train)
ax[1].set_title('Actual Training Labels')

plt.show()

# RPCA
from sklearn.decomposition import RandomizedPCA
randomized_rpca = RandomizedPCA(n_components=2)
reduce_data_rpca = randomized_rpca.fit_transform(X_train)
fig, ax = plt.subplots(1, 2, figsize=(8, 4))

fig.suptitle('Predicted Versus Training Labels(PCA)', fontsize=14, fontweight='bold')
fig.subplots_adjust(top=0.85)

ax[0].scatter(reduce_data_rpca[:, 0], reduce_data_rpca[:, 1], c=clusters)
ax[0].set_title('Predicted Training Labels')
ax[1].scatter(reduce_data_rpca[:, 0], reduce_data_rpca[:, 1], c=Y_train)
ax[1].set_title('Actual Training Labels')

plt.show()

