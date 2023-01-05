# PROGRAM 1 ################################################################

def aStarAlgo(start_node, stop_node):
    open_set = set(start_node)
    closed_set = set()
    g = {}              
    parents = {}         
    g[start_node] = 0
    parents[start_node] = start_node

    while len(open_set) > 0:
        n = None
        for v in open_set:
            if n == None or g[v] + heuristic(v) < g[n] + heuristic(n):
                n = v
        if n == stop_node or Graph_nodes[n] == None:
            pass
        else:
            for (m, weight) in get_neighbors(n):
                
                if m not in open_set and m not in closed_set:
                    open_set.add(m)
                    parents[m] = n
                    g[m] = g[n] + weight
                
                else:
                    if g[m] > g[n] + weight:
                        g[m] = g[n] + weight
                        parents[m] = n
                        
                        if m in closed_set:
                            closed_set.remove(m)
                            open_set.add(m)
                
        if n == None:
            print('Path does not exist!')
            return None
        
        if n == stop_node:
            path = []
            while parents[n] != n:
                path.append(n)
                n = parents[n]
            path.append(start_node)
            path.reverse()
            print('Path found: {}'.format(path))
            return path
      
        open_set.remove(n)
        closed_set.add(n)
    print('Path does not exist!')
    return None


def get_neighbors(v):
    if v in Graph_nodes:
        return Graph_nodes[v]
    else:
        return None

#for simplicity we ll consider heuristic distances given
#and this function returns heuristic distance for all nodes
def heuristic(n):
    H_dist = {'S': 5,'A': 4,'B':5, 'E':0}
    return H_dist[n]

Graph_nodes = {
    'S': [('A', 1), ('B', 2)],
    'A': [('E', 13)],
    'B': [('E',5)]}

aStarAlgo('S', 'E')
print("############################################################################")

############################################################################
# PROGRAM 2 ################################################################

class Graph:
    
    def __init__(self, graph, heuristicNodeList, startNode):  
        self.graph = graph
        self.H=heuristicNodeList
        self.start=startNode
        self.parent={}
        self.status={}
        self.solutionGraph={}
     
    def applyAOStar(self):         
        self.aoStar(self.start, False)

    def getNeighbors(self, v):     
        return self.graph.get(v,'')
    
    def getStatus(self,v):         
        return self.status.get(v,0)
    
    def setStatus(self,v, val):    
        self.status[v]=val
    
    def getHeuristicNodeValue(self, n):
        return self.H.get(n,0)     
 
    def setHeuristicNodeValue(self, n, value):
        self.H[n]=value           
        
    
    def printSolution(self):
        print("FOR GRAPH SOLUTION,TRAVERSE GRAPH FROM START NODE:",self.start)
        print("------------------------------------------------------------")
        print(self.solutionGraph)
        print("------------------------------------------------------------")
        
    def computeMinimumCostChildNodes(self, v):  
        minimumCost=0
        costToChildNodeListDict={}
        costToChildNodeListDict[minimumCost]=[]
        flag=True
        for nodeInfoTupleList in self.getNeighbors(v): 
            cost=0
            nodeList=[]
            for c, weight in nodeInfoTupleList:
                cost=cost+self.getHeuristicNodeValue(c)+weight
                nodeList.append(c)
            
            if flag==True:                       
                minimumCost=cost
                costToChildNodeListDict[minimumCost]=nodeList 
                flag=False
            else:  
                if minimumCost>cost:
                    minimumCost=cost
                    costToChildNodeListDict[minimumCost]=nodeList 
            
        return minimumCost, costToChildNodeListDict[minimumCost] 
    
    
    def aoStar(self, v, backTracking):     
        print("HEURISTIC VALUES  :", self.H)
        print("SOLUTION GRAPH    :", self.solutionGraph)
        print("PROCESSING NODE   :", v)
        print("-----------------------------------------------------------------------------------------")
        
        if self.getStatus(v) >= 0:       
            
            minimumCost, childNodeList = self.computeMinimumCostChildNodes(v)
            self.setHeuristicNodeValue(v, minimumCost)
            self.setStatus(v,len(childNodeList))
            
            solved=True  
            
            for childNode in childNodeList:
                self.parent[childNode]=v
                if self.getStatus(childNode)!=-1:
                    solved=solved & False
         
            if solved==True:             
                self.setStatus(v,-1)    
                self.solutionGraph[v]=childNodeList 
               
            if v!=self.start:  
                self.aoStar(self.parent[v], True)
                
            if backTracking==False:     
                for childNode in childNodeList: 
                    self.setStatus(childNode,0) 
                    self.aoStar(childNode, False) 
        
                                       
h1 = {'A': 9, 'B': 3, 'C': 4, 'D': 5, 'E': 5, 'F': 7, 'G': 4, 'H': 4}
graph1 = {
    'A': [[('B', 1), ('C', 1)], [('D', 1)]],
    'B': [[('E', 1)], [('F', 1)]],
    'D': [[('G', 1), ('H', 1)]]
      
}
G1= Graph(graph1, h1, 'A')
G1.applyAOStar()
G1.printSolution()


# PROGRAM 3 ################################################################

import pandas as pd
import numpy as np
data = pd.read_csv('enjoy.csv')

concept = np.array(data)[:,:-1]
target =np.array(data)[:,-1]
concept , target

def learn (concept,target):
    sp=concept[0].copy()
    print("spect ,general")
    print(sp)
    gn= [["?" for i in range(len(sp))] for i in range(len(sp))]
    print(gn)
    for i,h in enumerate(concept):
        if target[i]=="Yes":
            print("instance is positive")
            for x in range(len(sp)):
                if h[x]!=sp[x]:
                    sp[x]='?'
                    gn[x][x]='?'
        if target[i]=="No":
            print("instance is Negative")
            for x in range(len(sp)):
                if h[x]!=sp[x]:
                    gn[x][x]=sp[x]
                else:
                    gn[x][x]='?'
                    
        print("iteration ["+str(i+1)+"]")
        print("Spect : "+str(sp))
        print("gen :"+str(gn))
    gn=[gn[i] for i,h in enumerate(gn) if h!=["?" for x in range(len(sp))]]
    return gn,sp

gn,sp=learn(concept,target)

print("Final hypothesis: ")
print("Specific: "+str(sp))
print("General: "+str(gn))

print("############################################################################")
############################################################################
# PROGRAM 5 ################################################################


import numpy as np
X = np.array(([2, 9], [1, 5], [3, 6]), dtype=float)
y = np.array(([92], [86], [89]), dtype=float)
X = X/np.amax(X,axis=0)
y = y/100
def sigmoid (x):
    return 1/(1 + np.exp(-x))
def derivatives_sigmoid(x):
    return x * (1 - x)


epoch=1000
lr=0.1
inputlayer_neurons = 2
hiddenlayer_neurons = 3
output_neurons = 1 
wh=np.random.uniform(size=(inputlayer_neurons,hiddenlayer_neurons))
bh=np.random.uniform(size=(1,hiddenlayer_neurons))
wout=np.random.uniform(size=(hiddenlayer_neurons,output_neurons))
bout=np.random.uniform(size=(1,output_neurons))


for i in range(epoch):

    h_ip=np.dot(X,wh)+bh
    h_act = sigmoid(h_ip)
    o_ip=np.dot(h_act,wout)
    output = sigmoid(o_ip)

#     EO = y-output
#     outgrad = derivatives_sigmoid(output)
#     d_output = EO* outgrad

#     EH = d_output.dot(wout.T)
#     hiddengrad = derivatives_sigmoid(h_act)
#     d_hidden = EH * hiddengrad
    
    d_output = (y-output) * derivatives_sigmoid(output)
    d_hidden = d_output.dot(wout.T) * derivatives_sigmoid(h_act)

    wout += h_act.T.dot(d_output) *lr
    wh += X.T.dot(d_hidden) *lr
    
print("Input: \n" + str(X))
print("Actual Output: \n" + str(y))
print("Predicted Output: \n" ,output)

print("############################################################################")
############################################################################
# PROGRAM 6 ################################################################

import pandas as pd
data =pd.read_csv('data.csv')
data.describe()

from sklearn.model_selection import train_test_split
x=data.drop('Outcome',axis=1)
y=data[['Outcome']]

x_train,x_test,y_train,y_test=train_test_split(x,y,
                                               test_size=0.30,random_state=1)

from sklearn.naive_bayes import GaussianNB
model=GaussianNB()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)

from sklearn import metrics
print("Accc :",metrics.accuracy_score(y_test,y_pred))

print("############################################################################")
############################################################################
# PROGRAM 7 ################################################################

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import datasets
import pandas as pd
import numpy as np

iris = datasets.load_iris()
x=pd.DataFrame(iris.data)
x.columns=['sl','sw','pl','pw']
y=pd.DataFrame(iris.target)
y.columns=['target']

model=KMeans(n_clusters=3)
model.fit(x)
plt.figure(figsize=(7,7))
colormap = np.array(['red','lime','green'])

plt.subplot(2,2,1)
plt.scatter(x.pl,x.pw,c=colormap[y.target],s=40)
plt.title("Real cluster")
plt.xlabel("Petal lendth")
plt.ylabel("Petak width ")

plt.subplot(2,2,2)
plt.scatter(x.pl,x.pw,c=colormap[model.labels_],s=40)
plt.title("Kmeans cluster")
plt.xlabel("Petal lendth")
plt.ylabel("Petak width ")
plt.show()

#--------------------------------------------------------------

from sklearn import preprocessing
scaler =preprocessing.StandardScaler()
scaler.fit(x)
xsa=scaler.transform(x)
xs=pd.DataFrame(xsa,columns=x.columns)

from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components=3)
gmm.fit(xs)
gmm_y=gmm.predict(xs)

plt.subplot(1,2,2)
plt.scatter(x.pl,x.pw,c=colormap[gmm_y],s=40)
plt.title("GMM clustering")
plt.xlabel("Petal lendth")
plt.ylabel("Petak width ")
plt.show()

print("############################################################################")
############################################################################
# PROGRAM 8 ################################################################

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets

iris = datasets.load_iris()
x_train,x_test,y_train,y_test=train_test_split(iris.data,
                                               iris.target, test_size=0.1)

print("size of training data and its label",x_train.shape, y_train.shape)
print("size of testing data and its label",x_test.shape, y_test.shape)

for i in range(len(iris.target_names)):
    print("label",i, "-",str(iris.target_names[i]))

classifier=KNeighborsClassifier(n_neighbors=1)
classifier.fit(x_train,y_train)
y_pred=classifier.predict(x_test)

for r in range(0,len(x_test)):
    print("Sample:",str(x_test[r]), "Actual-label:",str(y_test[r]), 
          "Predicted-label:",str(y_pred[r]))
print("classification accuracy:", classifier.score(x_test,y_test))

print("############################################################################")
############################################################################

# PROGRAM 9 ################################################################

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def kernel(point, xmat, k):
    m,n = np.shape(xmat)
    weights = np.mat(np.eye((m)))
    for j in range(m):
        diff = point - X[j]
        weights[j,j] = np.exp(diff*diff.T/(-2.0*k**2))
    return weights

def localWeight(point, xmat, ymat, k):
    wei = kernel(point,xmat,k)
    W = (X.T*(wei*X)).I*(X.T*(wei*ymat.T))
    return W

def localWeightRegression(xmat, ymat, k):
    m,n = np.shape(xmat)
    y_pred = np.zeros(m)
    for i in range(m):
        y_pred[i] = xmat[i]*localWeight(xmat[i],xmat,ymat,k)
    return y_pred

def graphPlot(X,y_pred):
    sortindex = X[:,1].argsort(0)
    xsort = X[sortindex][:,0]
    #fig = plt.figure()
    #ax = fig.add_subplot(1,1,1)
    plt.subplot(1,1,1)
    plt.scatter(bill,tip, color='green')
    plt.plot(xsort[:,1],y_pred[sortindex], color = 'red', linewidth=5)
    plt.xlabel('Total bill')
    plt.ylabel('Tip')
    plt.show()

data = pd.read_csv('bill.csv')
bill = np.array(data.total_bill)
tip = np.array(data.tip)
mbill = np.mat(bill)
mtip = np.mat(tip)
m= np.shape(mbill)[1]
one = np.mat(np.ones(m))
X = np.hstack((one.T,mbill.T))
y_pred = localWeightRegression(X,mtip,0.5)

graphPlot(X,y_pred)


############################################################################
