import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
import pickle
X, y = load_svmlight_file( "../train", multilabel = False, n_features = 225, offset = 0 )
# X= X.toarray(order=None, out=None)
# NOTE:not much improvement detected when test size is 0.9 or 0.2 of original
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
# print(np.size(X_train,0))
# print(np.size(X_train,1))
# print(np.size(X_test,0))
# print(np.size(X_test,1))
# print(np.size(y_train,0))
clf = LogisticRegression(max_iter=np.size(X_train,0),multi_class='ovr',random_state=1).fit(X_train, y_train)

filename = 'model.npz'
print('done')
pickle.dump(clf, open(filename, 'wb'))

# NOTE: not much change when changing maxi_ter

# print(res.shape)
# print(res)
# print(y_test)
# sum =0
# for i in range(np.size(res,0)):
#     sum += (res[i] == y_test[i])

# print(sum/np.size(res,0)*100)







