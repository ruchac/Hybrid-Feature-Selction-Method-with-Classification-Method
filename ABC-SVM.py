# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 05:06:31 2019

@author: rucha
"""

Artificial Bee Colony Algorithm : 
# Required Libraries
import pandas as pd
import numpy  as np
import random
import os

# Function: Fitness Value
def fitness_calc (function_value):
    if(function_value >= 0):
        fitness_value = 1.0/(1.0 + function_value)
    else:
        fitness_value = 1.0 + abs(function_value)
    return fitness_value

# Function: Fitness Matrix
def fitness_matrix_calc(sources):
    fitness_matrix = sources.copy(deep = True)
    for i in range(0, fitness_matrix.shape[0]):
        function_value = target_function(fitness_matrix.iloc[i,0:fitness_matrix.shape[1]-2])
        fitness_matrix.iloc[i,-2] = function_value
        fitness_matrix.iloc[i,-1] = fitness_calc(function_value)
    return fitness_matrix

# Function: Initialize Variables
def initial_sources (food_sources = 3, min_values = [-5,-5], max_values = [5,5]):
    sources = pd.DataFrame(np.zeros((food_sources, len(min_values))))
    sources['Function'] = 0.0
    sources['Fitness' ] = 0.0
    for i in range(0, food_sources):
        for j in range(0, len(min_values)):
            sources.iloc[i,j] = random.uniform(min_values[j], max_values[j])
            #sources.iloc[i,j] = np.random.normal(0, 1, 1)[0]
    return sources

# Function: Employed Bee
def employed_bee(fitness_matrix, min_values = [-5,-5], max_values = [5,5]):
    searching_in_sources = fitness_matrix.copy(deep = True)
    new_solution = pd.DataFrame(np.zeros((1, fitness_matrix.shape[1] - 2)))
    trial        = pd.DataFrame(np.zeros((fitness_matrix.shape[0], 1)))
    for i in range(0, searching_in_sources.shape[0]):
        phi = random.uniform(-1, 1)
        j   = np.random.randint(searching_in_sources.shape[1] - 2, size = 1)[0]
        k   = np.random.randint(searching_in_sources.shape[0], size = 1)[0]
        while i == k:
            k = np.random.randint(searching_in_sources.shape[0], size=1)[0]
        xij = searching_in_sources.iloc[i, j]
        xkj = searching_in_sources.iloc[k, j]
        vij = xij + phi*(xij - xkj)
        
        for variable in range(0, searching_in_sources.shape[1] - 2):
            new_solution.iloc[0, variable] = searching_in_sources.iloc[i, variable]
        new_solution.iloc[0, j] = vij
        if (new_solution.iloc[0, j] > max_values[j]):
            new_solution.iloc[0, j] = max_values[j]
        elif(new_solution.iloc[0, j] < min_values[j]):
            new_solution.iloc[0, j] = min_values[j]
            
        new_function_value = float(target_function(new_solution.iloc[0,0:new_solution.shape[1]]))
        
        new_fitness = fitness_calc(new_function_value)
        
        if (new_fitness > searching_in_sources.iloc[i,-1]):
            searching_in_sources.iloc[i,j]  = new_solution.iloc[0, j]
            searching_in_sources.iloc[i,-2] = new_function_value
            searching_in_sources.iloc[i,-1] = new_fitness
        else:
            trial.iloc[i,0] = trial.iloc[i,0] + 1
        
        for variable in range(0, searching_in_sources.shape[1] - 2):
            new_solution.iloc[0, variable] = 0.0
            
    return searching_in_sources, trial

# Function: Probability Matrix
def probability_matrix(searching_in_sources):
    probability_values = pd.DataFrame(0, index = searching_in_sources.index, columns = ['probability','cumulative_probability'])
    source_sum = searching_in_sources['Fitness'].sum()
    for i in range(0, probability_values.shape[0]):
        probability_values.iloc[i, 0] = searching_in_sources.iloc[i, -1]/source_sum
    probability_values.iloc[0, 1] = probability_values.iloc[0, 0]
    for i in range(1, probability_values.shape[0]):
        probability_values.iloc[i, 1] = probability_values.iloc[i, 0] + probability_values.iloc[i - 1, 1]  
    return probability_values

# Function: Select Next Source
def source_selection(probability_values):
    random = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
    source = 0
    for i in range(0, probability_values.shape[0]):
        if (random <= probability_values.iloc[i, 1]):
          source = i
          break     
    return source

def outlooker_bee(searching_in_sources, probability_values, trial, min_values = [-5,-5], max_values = [5,5]):
    improving_sources = searching_in_sources.copy(deep = True)
    new_solution = pd.DataFrame(np.zeros((1, searching_in_sources.shape[1] - 2)))
    trial_update = trial.copy(deep = True)
    for repeat in range(0, improving_sources.shape[0]):
        i = source_selection(probability_values)
        phi = random.uniform(-1, 1)
        j   = np.random.randint(improving_sources.shape[1] - 2, size=1)[0]
        k   = np.random.randint(improving_sources.shape[0], size=1)[0]
        while i == k:
            k = np.random.randint(improving_sources.shape[0], size=1)[0]
        xij = improving_sources.iloc[i, j]
        xkj = improving_sources.iloc[k, j]
        vij = xij + phi*(xij - xkj)
        
        for variable in range(0, improving_sources.shape[1] - 2):
            new_solution.iloc[0, variable] = improving_sources.iloc[i, variable]
        new_solution.iloc[0, j] = vij
        if (new_solution.iloc[0, j] > max_values[j]):
            new_solution.iloc[0, j] = max_values[j]
        elif(new_solution.iloc[0, j] < min_values[j]):
            new_solution.iloc[0, j] = min_values[j]
        new_function_value = float(target_function(new_solution.iloc[0,0:new_solution.shape[1]]))
        new_fitness = fitness_calc(new_function_value)
        
        if (new_fitness > improving_sources.iloc[i,-1]):
            improving_sources.iloc[i,j]  = new_solution.iloc[0, j]
            improving_sources.iloc[i,-2] = new_function_value
            improving_sources.iloc[i,-1] = new_fitness
            trial_update.iloc[i,0] = 0
        else:
            trial_update.iloc[i,0] = trial_update.iloc[i,0] + 1
        
        for variable in range(0, improving_sources.shape[1] - 2):
            new_solution.iloc[0, variable] = 0.0    
    return improving_sources, trial_update

def scouter_bee(improving_sources, trial_update, limit = 3):
    for i in range(0, improving_sources.shape[0]):
        if (trial_update.iloc[i,0] > limit):
            for j in range(0, improving_sources.shape[1] - 2):
                improving_sources.iloc[i,j] = np.random.normal(0, 1, 1)[0]
            function_value = target_function(improving_sources.iloc[i,0:improving_sources.shape[1]-2])
            improving_sources.iloc[i,-2] = function_value
            improving_sources.iloc[i,-1] = fitness_calc(function_value)

    return improving_sources

# ABC Function
def artificial_bee_colony_optimization(food_sources = 3, iterations = 50, min_values = [-5,-5], max_values = [5,5], employed_bees = 3, outlookers_bees = 3, limit = 3):  
    count = 0
    best_value = float("inf")
    sources = initial_sources(food_sources = food_sources, min_values = min_values, max_values = max_values)
    fitness_matrix = fitness_matrix_calc(sources)
    
    while (count <= iterations):
        print("Iteration = ", count, " f(x) = ", best_value)
       
        e_bee = employed_bee(fitness_matrix, min_values = min_values, max_values = max_values)
        for i in range(0, employed_bees - 1):
            e_bee = employed_bee(e_bee[0], min_values = min_values, max_values = max_values)
        probability_values = probability_matrix(e_bee[0])
            
        o_bee = outlooker_bee(e_bee[0], probability_values, e_bee[1], min_values = min_values, max_values = max_values)
        for i in range(0, outlookers_bees - 1):
            o_bee = outlooker_bee(o_bee[0], probability_values, o_bee[1], min_values = min_values, max_values = max_values)

        if (best_value > o_bee[0].iloc[o_bee[0]['Function'].idxmin(),-2]):
            best_solution = o_bee[0].iloc[o_bee[0]['Function'].idxmin(),:].copy(deep = True)
            best_value = o_bee[0].iloc[o_bee[0]['Function'].idxmin(),-2]
       
        sources = scouter_bee(o_bee[0], o_bee[1], limit = limit)
        fitness_matrix = fitness_matrix_calc(sources)
        
        count = count + 1   
    print(best_solution[0:len(best_solution)-1])
    return best_solution[0:len(best_solution)-1]

SVM Function : 

from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
    
# Function to be Minimized
def target_function(variables_values = [0.5,0.5]):
    func_value = 0
    C = 1.0
    SVMRegr = SVC()
    Dataset = pd.read_csv('abc.csv')
    X = Dataset.drop(['pqr'],axis=1)
    y = Dataset.iloc[:,30].values
    penalty = 0.02
    colname = X.columns
    penalty = 0.02
    for i in range(0, len(variables_values)):
        if variables_values[i] < 0.5 and X.shape[1] > 1:
            X =  X.drop(colname[i], axis = 1)
    le = LabelEncoder()
    X['a']= le.fit_transform(X['a'])
    X['b']= le.fit_transform(X['b'])
    y= le.fit_transform(y)
    from sklearn.cross_validation import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20,random_state=2,stratify=y)
    from sklearn import preprocessing
    Xscaled_train = pd.DataFrame(preprocessing.scale(X_train.values))
    Xscaled_test  =  pd.DataFrame(preprocessing.scale(X_test.values))
    from sklearn import svm
    svc = svm.SVC(kernel='linear',random_state=0)
    svc.fit(Xscaled_train,y_train)
    y_pred = svc.predict(Xscaled_test)
    from sklearn.metrics import classification_report , confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    from sklearn.metrics import accuracy_score
    acc=accuracy_score(y_test, y_pred)
    func_value = 1 - acc + penalty*(X.shape[1])
    return func_value

abc = artificial_bee_colony_optimization(food_sources = 20, iterations = 5, min_values = [0]*34, max_values = [1]*34, employed_bees = 20, outlookers_bees = 20, limit = 80)
# Results
def results(variables_values = [0, 0]):
    SVMRegr = SVC()
    Dataset = pd.read_csv('abc.csv')
    X = Dataset.drop(['pqr'],axis=1)
    y = Dataset.iloc[:,30].values
    colname = X.columns
    df = pd.DataFrame(np.zeros((X.shape[1], 2)), columns=['Variables', 'Importance'])
    for i in range(0, len(variables_values)):
        df.iloc[i,0] = colname[i]
        df.iloc[i,1] = variables_values[i]
    for i in range(0, len(variables_values)):
        if variables_values[i] < 0.50 and X.shape[1] > 1:
            X =  X.drop(colname[i], axis = 1)    
            df = df.drop(df[df['Variables'] == colname[i]].index)
    df = df.sort_values(by = ['Importance'], ascending = False)
    SVMRegr.fit(Xscaled_train,y_train)
    y_pred = svc.predict(Xscaled_test)
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    return df,X

from sklearn.metrics import accuracy_score
result=accuracy_score(y_test, y_pred)
[X_train,acc],variables = results(variables_values = abc[0:len(abc)-1])
print('Accuracy = ', str(result)) 
print(variables)

#GET ROC DATAfrom sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve( y_pred,y_test)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()
