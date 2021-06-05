#Importing the datasets
import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation

from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from IPython.display import HTML


#loading the dataset
boston=load_boston()

#description of the dataset
print(boston.DESCR)

features = pd.DataFrame(boston.data,columns=boston.feature_names)
print(features)

print(features[['AGE','TAX']])
print(features.iloc[:,0:2])

target = pd.DataFrame(boston.target,columns=['target'])
print(target)

df=pd.concat([features,target],axis=1)
print(df)

print(df.describe().round(decimals=2))

#creating a correlation object for the df dataframe
corr=df.corr('pearson')

#finding the correlation factor for each of the attribute in the features with the target vaule 
corrs=[abs(corr[attr]['target']) for attr in list(features)]

#make a list of pairs [(corr,feature)]
l=list(zip(corrs,list(features)))

#sorting the list of (corr,feature) in descending order(reverse=True) based on corrs(key=lambda x:x[0])
l.sort(key=lambda x:x[0],reverse=True)

#unzip pairs to two lists 
#zip(*l) takes a list that looks like [[a,b,c],[d,e,f],[g,h,i]]
#and gives[[a,d,g],[b,e,h],[c,f,i]]
corrs,labels=list(zip((*l)))

#plotting the correlations w.r.t target values as bar graphs
index=np.arange(len(labels))
plt.figure(figsize=(15,5))
plt.bar(index,corrs,width=0.5)
plt.xlabel("Attributes")
plt.ylabel("correlation with the target variable")
plt.xticks(index,labels)
plt.show()

#obtaining the max correlation value and storing in X
X=df["LSTAT"].values
Y=df["target"].values

#scaling tha data
x_scaler = MinMaxScaler()
X=x_scaler.fit_transform(X.reshape(-1,1))
X=X[:,-1]
y_scaler = MinMaxScaler()
Y=y_scaler.fit_transform(Y.reshape(-1,1))
Y=Y[:,-1]

#splitting the data
xtrain,xtest,ytrain,ytest = train_test_split(X,Y,test_size=0.2)

#Error Function
def error(m,x,c,t):
    N=x.size
    e=sum(((m*x+c)-t)**2)
    return e*1/(2*N)

#update function
def update(m,x,c,t,learning_rate):
    grad_m = sum( 2 * ((m * x + c)- t) * x)          
    grad_c = sum(2 * ((m*x+c)-t))
    m=m-grad_m*learning_rate       
    c=c-grad_c*learning_rate
    return m,c

#gradient descent function
def gradient_descent(init_m,init_c,x,t,learning_rate,iterations,error_threshold):
    m=init_m
    c=init_c
    error_values=list()
    mc_values=list()
    for i in range(iterations):
        e=error(m,x,c,t)
        if e<error_threshold:
            print("Error less than Threshold... stopping gradient descent")
            break
        error_values.append(e)
        m,c=update(m,x,c,t,learning_rate)
        mc_values.append((m,c))
    return m,c,error_values,mc_values

#obtaining the final m and c values
init_m=0.9
init_c=0
learning_rate=0.001   #should be less than 0.0025
iterations=250
error_threshold=0.001
m,c,error_values,mc_values=gradient_descent(init_m,init_c,xtrain,ytrain,learning_rate,iterations,error_threshold)

#As the number of iterations increases, changes in the line are less noticeable
#In order to reduce the processing time for the animation, it si advised to choose smaller values
mc_values_anim=mc_values[0:250:5]


#visualizing the model training
fig,ax=plt.subplots()
ln,=plt.plot([],[],'ro-',animated=True)

def init():
    plt.scatter(xtest,ytest,color='g')
    ax.set_xlim(0,1.0)
    ax.set_ylim(0,1.0)
    return ln,

def update_frame(frame):
    m,c=mc_values_anim[frame]
    x1,y1=-0.5,m*-.5+c
    x2,y2=1.5,m*1.5+c
    ln.set_data([x1,x2],[y1,y2])
    return ln,

anim = FuncAnimation(fig, update_frame, frames=range(len(mc_values_anim)), init_func = init, blit = True)
HTML(anim.to_html5_video())

#plotting the actual train and predicted values
plt.scatter(xtrain,ytrain,color='g')
plt.plot(xtrain,(m*xtrain+c),color='r')

#plotting the error values 
plt.plot(np.arange(len(error_values)),error_values)
plt.ylabel('Error')
plt.xlabel('Iterations')

predicted = (m*xtest)+c

print("Mean_squared_error:\n",mean_squared_error(ytest, predicted))

#placing xtest, ytest, Predicted side by side using dataframe
p=pd.DataFrame(list(zip(xtest,ytest,predicted)),columns=['x','target values','predicted'])
print("predicted house price values for test dataset before scaling:\n",p.head())

plt.scatter(xtest,ytest,color='g')
plt.plot(xtest,predicted,color='b')

#predicting for xtest data after removing scaling in order to obtain the actual price
predicted=predicted.reshape(-1,1)
xtest=xtest.reshape(-1,1)
ytest=ytest.reshape(-1,1)

xtest_scaled = x_scaler.inverse_transform(xtest)
ytest_scaled = y_scaler.inverse_transform(ytest)
predicted_scaled = y_scaler.inverse_transform(predicted)

#this is to remove the extra dimension created during scaling
xtest_scaled=xtest_scaled[:,-1]
ytest_scaled=ytest_scaled[:,-1]
predicted_scaled=predicted_scaled[:,-1]

p=pd.DataFrame(list(zip(xtest_scaled,ytest_scaled,predicted_scaled)),columns=['x','Target values','Predicted values'])
p=p.round(decimals=2)
print("predicted house price values for test dataset:\n",p.head(10))