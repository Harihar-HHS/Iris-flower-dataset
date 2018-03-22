import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy import stats
from sklearn.datasets import load_iris
from sklearn import neighbors, datasets

data = load_iris()

print("\t\t\t\t\tIris Flower dataset prediction\n")

print ('\nKeys:', data.keys())                              # Key values for the dictionary data structures
print ('-' * 20)
print ('Data Shape:', data.data.shape)                      # no. of rows & columns
print ('-' * 20)
print ('Features:', data.feature_names)                     # features of the flower
print ('-' * 20)

#Petal Length vs Sepal Width
plt.scatter(data.data[:, 1], data.data[:, 2], c=data.target, cmap=plt.cm.get_cmap('Set1', 3))
plt.xlabel(data.feature_names[1])
plt.ylabel(data.feature_names[2])

color_bar_formating = plt.FuncFormatter(lambda i, *args: data.target_names[int(i)])
plt.colorbar(ticks = [0,1,2], format = color_bar_formating)

#Petal Length vs Sepal Width
plt.scatter(data.data[:, 2], data.data[:, 3], c=data.target, cmap=plt.cm.get_cmap('Set1', 3))
plt.xlabel(data.feature_names[2])
plt.ylabel(data.feature_names[3])

color_bar_formating = plt.FuncFormatter(lambda i, *args: data.target_names[int(i)])
plt.colorbar(ticks = [0,1,2], format = color_bar_formating)


# where X = measurements and y = species
X, y = data.data, data.target

#define the model
knn = neighbors.KNeighborsClassifier(n_neighbors=5, weights='uniform')              # k neighbours model is used because it gives the most accurate prediction compared to other models

#fit/train the model
knn.fit(X,y)

#What species has a (('a' cm) x ('b' cm)) sepal and a (('c' cm) x ('d' cm) petal?
a=float(input("\nsepal length : "))                     # user input
b=float(input("sepal width : "))
c=float(input("petal length : "))
d=float(input("petal width : "))
X_pred = [a,b,c,d]                                      # test data
output = knn.predict([X_pred])                          # predicting using the model created

# output
print ('\nPredicted Species:', data.target_names[output])
print ('Options:', data.target_names)
print ('Probabilities:', knn.predict_proba([X_pred]))

input("\n\nPress any key to exit.....")
