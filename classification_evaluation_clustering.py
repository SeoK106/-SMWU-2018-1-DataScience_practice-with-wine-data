'''
Author: SeoK106
This is the code to practice the concept of data science through python and scikit-learn.
Especially It focused on classification, evaluation, and clustering.
'''

import numpy as np
from sklearn import tree,svm,linear_model,neighbors
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score
from sklearn.cluster import AgglomerativeClustering,KMeans
from sklearn.model_selection import train_test_split

def print_menu():
    print("----------------------- MENU -----------------------")
    print("ML Tutorial program - Predict the Quality of Red wine\n")
    print("  1. Predict wine quality\n  2. Evaluate wine prediction models\n  3. Cluster wines\n  4. Quit")
    print("-"*52)

def get_data():
    '''Data source

    P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.
    Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.

    - inputs: ['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']
    - output: quality (score between 0 and 10)

    Web Link: http://archive.ics.uci.edu/ml/datasets/wine+quality
    '''
    
    data_path = '.\winequality-red.csv' # Edit the path
    data = np.genfromtxt(data_path,dtype=np.float32,delimiter=";",skip_header=1)

    x = data[:,0:11]
    y = data[:,11]

    # 80% for train data, 20% for test data 
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

    return x_train,x_test,y_train,y_test
    

def input_values():
    test_data = []
    data_types = ['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']

    print("\nInput 11 values of a wine\n")
    for i in range(len(data_types)):
        value = input(f"{i+1}. {data_types[i]}: ")
        if value:
            #test_data.append(value)
            test_data.insert(i,value)
        else:
            test_data.insert(i,0)

    test_data=np.array(test_data)
    return test_data
    
def print_quality_prediction():
    
    global x_train,x_test,y_train,y_test
    input_data = []
    input_data.append(input_values())
    
    print("\npredicted wine quality(0~10)\n")
    print("1. Decesion tree: ", int(DecisionTree(x_train,y_train,input_data)))
    print("2. Support vector machine: ", int(SVM(x_train,y_train,input_data)))
    print("3. Logostic regression: ", int(LogisticRegression(x_train,y_train,input_data)))
    print("4. k-NN classifier: ", int(kNN(x_train,y_train,input_data)))
    print()

# Geneatae models and Return prediction

def DecisionTree(x,y,sample):
    dtc = tree.DecisionTreeClassifier(random_state = 0)
    model = dtc.fit(x,y)
    predicted_target = model.predict(sample)
    return predicted_target    
    
def SVM(x,y,sample):
    svmc = svm.SVC(random_state = 0)
    model = svmc.fit(x,y)
    predicted_target = model.predict(sample)
    return predicted_target
    
def LogisticRegression(x,y,sample):
    lrc = linear_model.LogisticRegression(random_state = 0)
    model = lrc.fit(x,y)
    sample = np.array(sample,dtype=np.float64)
    predicted_target = model.predict(sample)
    return predicted_target
    
def kNN(x,y,sample):
    knnc = neighbors.KNeighborsClassifier(n_neighbors=5)
    model = knnc.fit(x,y)
    predicted_target = model.predict(sample)
    return predicted_target


# Evaluate model and Return the scores 

def evaluating_classification_models():

    '''
    Evaluate the performance of classification models
    We calculate
        1. confusion matrix
        2. accuracy
        3. precision
        4. recall
        5. f1 score
    '''

    global x_train,x_test,y_train,y_test
    y_true = y_test
 
    print("\nDecesion tree:")
    y_pred = DecisionTree(x_train,y_train,x_test)
    print_eval(y_true,y_pred)

   
    print("Support vector machine:")
    y_pred = SVM(x_train,y_train,x_test)
    print_eval(y_true,y_pred)
  

    print("Logostic regression:")
    y_pred = LogisticRegression(x_train,y_train,x_test)
    print_eval(y_true,y_pred)


    print("k-NN classifier:")
    y_pred = kNN(x_train,y_train,x_test)
    print_eval(y_true,y_pred)

def print_eval(y_true,y_pred):
    # "labels=np.unique(y_pred)" to avoid the case when some labels in y_test don't appear in y_pred (warning case)
    print(f"1. confusion matrix:\n {confusion_matrix(y_true,y_pred)}")
    print(f"2. accuracy: {accuracy_score(y_true,y_pred)}")
    print(f"3. precision: {precision_score(y_true,y_pred,average=None,labels=np.unique(y_pred))}")
    print(f"4. recall: {recall_score(y_true,y_pred,average=None,labels=np.unique(y_pred))}")
    print(f"5. f1 score: {f1_score(y_true,y_pred,average=None,labels=np.unique(y_pred))}")
    print()


# Generate clusters and display the number of wines in the same cluster

def print_clusters():
    data_path = '.\winequality-red.csv' # Edit the path
    data = np.genfromtxt(data_path,dtype=np.float32,delimiter=";",skip_header=1)

    x = data[:,0:11]   
    n_cluster = int(input("\nInput the number of clusters: "))
    
    print("\nThe number of wines in each cluster:\n")
    print("<hierarchical clustering>\n")
    h_cluster = hierc_clusteirng(x, n_cluster)
    for i in range(n_cluster):
        print(f"cluster {i+1}: {h_cluster[i]}")
    print(f"total:{sum(h_cluster)}")
        
    print("\n<k-means clustering>\n")
    km_cluster = kmeans_clustering(x,n_cluster)
    for i in range(n_cluster):
        print(f"cluster {i+1}: {km_cluster[i]}")
    print(f"total:{sum(km_cluster)}\n")
    
def hierc_clusteirng(data,n_cluster):
    x = np.array(data)
    model = AgglomerativeClustering(n_cluster)
    model.fit(x)
    
    return count_clusters(n_cluster,model.labels_)
    
def kmeans_clustering(data,n_cluster):
    x = np.array(data)
    model = KMeans(n_cluster,random_state=0)
    model.fit(x)

    return count_clusters(n_cluster,model.labels_)

def count_clusters(n_cluster,label_data):
    temp = [0]*n_cluster
    for label in label_data:
        temp[label] += 1
    return temp

if __name__=="__main__":
    x_train,x_test,y_train,y_test = get_data()
    while True:
        print_menu()
        menu=int(input(">> "))
        if menu==1:
            print_quality_prediction()
            continue
        if menu==2:
            evaluating_classification_models()
            continue
        if menu==3:
            print_clusters()
            continue
        if menu==4:
            break
