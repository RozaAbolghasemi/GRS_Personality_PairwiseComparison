#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 12:07:40 2020

@author: rozaabol
"""

import pandas as pd
import numpy as np
import csv
import os
#%% 
#Convert given data from txt to csv

if not os.path.isfile('Movies.csv'):
    with open('./Dataset/Movies.txt', 'r') as in_file:
        stripped = (line.strip() for line in in_file)
        lines = (line.split(";") for line in stripped if line)
        with open('Movies.csv', 'w') as out_file:
            writer = csv.writer(out_file)
            writer.writerow(('MovieID', 'title','Link'))
            writer.writerows(lines)

if not os.path.isfile('Ratings.csv'):
    with open('./Dataset/Ratings.txt', 'r') as in_file:
        stripped = (line.strip() for line in in_file)
        lines = (line.split(";") for line in stripped if line)
        with open('Ratings.csv', 'w') as out_file:
            writer = csv.writer(out_file)
            writer.writerow(('UserID', 'MovieID','Rating'))
            writer.writerows(lines)
    
if not os.path.isfile('Comparisons.csv'):
    with open('./Dataset/Comparisons.txt', 'r') as in_file:
        stripped = (line.strip() for line in in_file)
        lines = (line.split(";") for line in stripped if line)
        with open('Comparisons.csv', 'w') as out_file:
            writer = csv.writer(out_file)
            writer.writerow(('UserID', 'MovieID1','MovieID2','PairwiseScore'))
            writer.writerows(lines)
#%%        
Movies = pd.read_csv("./Movies.csv")#.to_numpy()
print ("Movies.shape" ,Movies.shape)
print ("Movies.title[2]", Movies.title[2])

Ratings = pd.read_csv("./Ratings.csv")#.to_numpy()
print ("Ratings.shape",Ratings.shape)

Comparisons = pd.read_csv("./Comparisons.csv")#.to_numpy()
print ("Comparisons.shape",Comparisons.shape)
#%%
#Convert ratings to pairwiseRates, Adding them to  comparison file and read and save as PairwiseRates
if not os.path.isfile('PairwiseRates.csv'): 
    with open(r'Comparisons.csv', 'a') as f:
        writer = csv.writer(f)
     
        ii,iii= Ratings.shape
        for i in range(ii):
            for j in range (i+1,ii):
                if Ratings.UserID[i]==Ratings.UserID[j]:
                    Comparison=Ratings.Rating[i]-Ratings.Rating[j]
                    fields=[Ratings.UserID[i],Ratings.MovieID[i],Ratings.MovieID[j], Comparison]
                    writer.writerow(fields)
                else:
                    break
    PairwiseRates = pd.read_csv("./Comparisons.csv")#.to_numpy()
    df=pd.DataFrame(PairwiseRates)
    df.to_csv('PairwiseRates.csv', header=True, index=False)  
    print (PairwiseRates.shape)
else:
    PairwiseRates = pd.read_csv("./PairwiseRates.csv")#.to_numpy()
    print ("PairwiseRates.shape",PairwiseRates.shape)
#%%
#Movie_pairs is a list containing paired movies
ii,iii= PairwiseRates.shape
Movie_pairs = []
for i in range(ii):
   Movie_pairs.append((PairwiseRates.MovieID1[i],PairwiseRates.MovieID2[i]))
unique_Movie_pairs = list(dict.fromkeys(Movie_pairs))

print("*")
#%%
# To make access to indexes easier, I used dictionaries:
UniqueUserIDs=np.unique(PairwiseRates.UserID)
unique_Movie_pairs 


dict_index2user={}
keys=range (len(UniqueUserIDs))
values=UniqueUserIDs
for i in keys:
    dict_index2user [values[i]]= i
print(dict_index2user[10])

dict_index2moviepairs={}
keys= range (len(unique_Movie_pairs))
values=unique_Movie_pairs
for i in keys:
    dict_index2moviepairs[values[i]]= i
print(dict_index2moviepairs[(2628,3176)])



print("**")


    
#%%
# Mtrix R:   Rows:unique users   Columns: Unique Movie pairs
    
R = np.ones([len(UniqueUserIDs),len(unique_Movie_pairs)])*10 # I itialized the matriz with 10 (a number which is not in the interval of -5 to 5)
for row in range (len(PairwiseRates)):
    u = dict_index2user[PairwiseRates.UserID[row]]
    i = dict_index2moviepairs[(PairwiseRates.MovieID1[row],PairwiseRates.MovieID2[row])]
    R[u][i]=PairwiseRates.PairwiseScore[row]

print("***")
    
#%%

# Matrix Factorization
import numpy as np

def mf(R, k, n_epoch=5000, lr=.0003, l2=.04): #n_epoch=5000, lr=.0003
  tol = .001  # Tolerant loss.
  m, n = R.shape
  R2=R-10 
  # Initialize the embedding weights.
  P = np.random.rand(m, k)
  Q = np.random.rand(n, k)
  for epoch in range(n_epoch):
    # Update weights by gradients.    
    for u, i in zip(*R2.nonzero()):
      err_ui = R[u,i] - P[u,:].dot(Q[i,:])
      for j in range(k):
        P[u][j] += lr * (2 * err_ui * Q[i][j] - l2/2 * P[u][j])
        Q[i][j] += lr * (2 * err_ui * P[u][j] - l2/2 * Q[i][j])
    print (epoch)
    # compute the loss.
    E = (R - P.dot(Q.T))**2
    obj = E[R.nonzero()].sum() + lr*((P**2).sum() +(Q**2).sum())
    if obj < tol:
        break
  return P, Q


#%%
k=10  #len of embeddings

UserEmbedding, MovieEmbedding= mf(R, k, n_epoch=100, lr=.003, l2=.04)


np.savetxt("UserEmbedding.csv", UserEmbedding, delimiter=",")
np.savetxt("MovieEmbedding.csv", MovieEmbedding, delimiter=",")

df=pd.DataFrame(UserEmbedding)
df.to_csv('UserEmbedding.csv', header=True, index=True) 

df=pd.DataFrame(MovieEmbedding)
df.to_csv('MovieEmbedding.csv', header=True, index=True) 


#%%

PairwiseRates = pd.read_csv("./PairwiseRates.csv")#.to_numpy()
UniqueUserIDs=np.unique(PairwiseRates.UserID)


UserEmbedding = pd.read_csv("./UserEmbedding.csv")#.to_numpy()
print ("UserEmbedding.shape" ,UserEmbedding.shape)

MovieEmbedding = pd.read_csv("./MovieEmbedding.csv")#.to_numpy()
print ("MovieEmbedding.shape",MovieEmbedding.shape)

UserEmbedding.pop("Unnamed: 0")
MovieEmbedding.pop("Unnamed: 0")
NewRating=UserEmbedding.dot(MovieEmbedding.transpose())

#%%  NewR is a Matrix containing new pairwise Rankings (as a result of matrix factorization)



PairwiseMatrix=UserEmbedding.dot(MovieEmbedding.transpose())

    
unique_MovieIDs=np.unique(PairwiseRates.MovieID1)

RatingMatrix = np.zeros((len(UniqueUserIDs), len(unique_MovieIDs)))
for movie in unique_MovieIDs:
    count = 0
    col =0
    for i in range (len(unique_Movie_pairs)): 
        if unique_Movie_pairs[i][0] == movie:
            count += 1
            for j in range (len(UniqueUserIDs)):
                RatingMatrix[j][col] +=  PairwiseMatrix[i][j] ###Bekhatere error jaye [i] va [j] dar PairwiseMatrix[i][j] ra avaz kardam.          
            col += 1
    for j in range (len(UniqueUserIDs)):
        RatingMatrix[j][col] /= count   
     
NewRating = RatingMatrix   
    


#%%
# Complete codes for different groups sizez from 2 to 20 (p kept fixed)

# GroupMemIDs contaning index of random users in the group


Num_Users, Num_Movies= NewRating.shape
Num_Mem=4  #number of group members
Num_Item= Num_Movies  #number of group items


GroupRanges= range(4,5)
FinalEvaluation = np.zeros((len(GroupRanges), 5))
GU_Score = [] #Difference between group score and user score
GPersonalities = []
#FE = pd.DataFrame(columns=['Group size', 'Precision', 'Recall', 'Fairness', 'Consensus']) 

for Num_Mem in GroupRanges: #V is number of group members #Repeat pricedure for different group sizes from 2 to 20.
    SumTP = 0   #sum of True Positive
    SumFP = 0   #sum of False Positive
    SumT = 0
    SumFairness = 0
    SumConsensus = 0
    for group in range (int(Num_Users/Num_Mem)):
        
        GroupMemIDs=list(range (group*Num_Mem,(group+1)*Num_Mem))
        #GroupMemIDs= np.random.randint(0, Num_Users-1, size=(Num_Mem))   #if you want to have random members
        #ItemIDs= np.random.randint(0, Num_Movies-1, size=(Num_Item))  #if you want to choose random items
        ItemIDs= np.arange(0, Num_Movies).tolist()
        
        
        # P is personality traits. How much a assertive or cooperative a user is.
        # W[i][j] is the strength of the influence of the jth expert on the ith one. which is calculated based on P
        
        p= np.random.rand(Num_Mem)   #If want to have random personality
        #p=  np.array([1.0,0.0,0.0,0.0])  #just an example, you can comment it  
        #p= np.ones(Num_Mem)             #If want to have equal personality
            
        
        
        w= np.empty([Num_Mem, Num_Mem], dtype=float)
        for i in range (Num_Mem):
          w[i][i]=1.0
          for j in range (Num_Mem):
            if i!=j:
              w[i][j]=p[j]/(p[i]+p[j])/(Num_Mem)
              w[i][i]-=w[i][j]
        #print("Weights>",w)
        
        
        ###GroupMemIDs=  np.array([0,1,2])  #just an example (3 first members), you can comment it 
        
        
        RR=np.empty(shape=(Num_Mem,Num_Item), dtype=float)
        
        j=0
        for g in GroupMemIDs:   ##Initial ratings
           k=0
           for i in ItemIDs:
              ###RR[i]=sum(NewRating[g][0:5])/5   #opinion on the first item(Average of pairwise items)... change it for more items ***************
              RR[j][k]=NewRating[g][i]   #bekhatere error inha [g][i] ra jabeja nakardam
              k+=1
           j+=1
        
        ###RR= np.array([5, 4, 1]).T#just an example you can comment it
        
        
        ###NewRates=np.dot(w, RR)
        #print("initial rates:", RR)
        
          
        #NewRates=np.dot(w, RR.transpose())
        #print(NewRates)
        
        import matplotlib.pyplot as plt
        %matplotlib inline 
        Iteration = 20
        NR=np.zeros([Iteration, Num_Mem, Num_Item], dtype=object)
        for i in range (Iteration):
          NewRates=np.dot(w, RR)
          NR[i][:][:]=RR
          RR=NewRates
          #print(i, NewRates)
         
        
        
        # Sorting rates in Descending order
        sort_index = np.argsort(NewRates[0][:]*-1) 
        BestItems_count=20
        BestItems=sort_index[0:BestItems_count]
        #print("BestItems", BestItems) 
        
         
        RG=BestItems

        test_data = NewRating
        
        
        # TP: True Positive
        TP=0
        Teta = 8
        for i in BestItems:
            for u in GroupMemIDs:
               if test_data[u][i]!=0 and test_data[u][i] >= Teta:
                  TP+=1
        SumTP += TP
        
        
        # FP: False positive
        FP=0
        for i in BestItems:
            if NewRates[0][i] <= Teta:  
               FP+=1
        SumFP += FP
        
        
        # T: Expected recommendations set
        T=0
        for i in range(Num_Movies):
            for u in GroupMemIDs:
               if test_data[u][i]!=0 and test_data[u][i] >= Teta:
                  T+=1
        SumT += T
        
        
        #Evaluation based on Consensus ans Fairness
        # Fairness is defined as the share of group members ui with at least m items in the recommended package for which ui has a high performance.
        
        SatisfiedMembers = 0
        Threshold=0.70
        for u in GroupMemIDs:
            F = 0
            for i in BestItems:
                if NewRating[u][i] >= Threshold:   #bekhatere error jaye [u] va [i] avaz nakardam.
                    F += 1
            if F > BestItems_count/2:
                SatisfiedMembers += 1
        SumFairness += SatisfiedMembers /Num_Mem      
           
        #Consensus is a measure of agreement between group members. Here it is pairwise distance between users final opinions.   
        SumC = 0
        for ui in range(Num_Mem):
            for uj in range(Num_Mem):
                SumC += NewRates[ui][0] - NewRates[uj][0]
        SumConsensus += 1 - SumC /  (Num_Mem * Num_Mem)    
        
        
        for i in p:
          GPersonalities.append(i)
        j=0
        for k in GroupMemIDs:
            GU_Score.append(NewRating[k][BestItems[0]]-NewRates[j][BestItems[0]])
            j+=1 
            
            
        
    Precision=SumTP/(SumTP+SumFP)
    #print("Precision:", Precision)
    #Recall=SumTP/SumT
    #print("Recall:", Recall)
    Fairness = SumFairness/ group
    #print ("Fairness:", Fairness) 
    Consensus = SumConsensus/ group
    #print ("Consensus:", Consensus)    
    
    
    FinalEvaluation[Num_Mem -2][0] = Num_Mem
    FinalEvaluation[Num_Mem -2][1] = Precision
    #FinalEvaluation[Num_Mem -2][2] = Recall
    FinalEvaluation[Num_Mem -2][3] = Fairness
    FinalEvaluation[Num_Mem -2][4] = Consensus
    
    

plt.scatter(GPersonalities, GU_Score)

#%%
    
#FinalEvaluation
df1 = pd.DataFrame(FinalEvaluation, columns = ['Group_size', 'Precision', 'Recall', 'Fairness', 'Consensus'])
   
df1.Group_size *= 10
 
plt.scatter('Precision', 'Fairness', 
             s='Group_size',
             alpha=0.5, 
             data=df1)
plt.xlabel("Precision", size=14)
plt.ylabel("Fairness", size=14)
plt.title("bubble sizes indicate Group size", size=14)








#%%







    
    
    
  #%%  

# Codes for one group size (Personality changes to see the results)

# GroupMemIDs contaning index of random users in the group


Num_Users, Num_Movies= NewRating.shape
Num_Mem=4  #number of group members
Num_Item= Num_Movies  #number of group items




SumTP = 0   #sum of True Positive
SumFP = 0   #sum of False Positive
SumT = 0
SumFairness = 0
SumConsensus = 0

FinalEvaluation = np.zeros((1, 5))
for group in range (int(Num_Users/Num_Mem)):
    
    GroupMemIDs=list(range (group*Num_Mem,(group+1)*Num_Mem))
    ItemIDs= np.arange(0, Num_Movies).tolist()
    
    
    w= np.empty([Num_Mem, Num_Mem], dtype=float)
    for i in range (Num_Mem):
      w[i][i]=1.0
      for j in range (Num_Mem):
        if i!=j:
          w[i][j]=p[j]/(p[i]+p[j])/(Num_Mem)
          w[i][i]-=w[i][j]
    #print("Weights>",w)
    
    
    ###GroupMemIDs=  np.array([0,1,2])  #just an example (3 first members), you can comment it 
    
    
    RR=np.empty(shape=(Num_Mem,Num_Item), dtype=float)
    
    j=0
    for g in GroupMemIDs:   ##Initial ratings
       k=0
       for i in ItemIDs:
          ###RR[i]=sum(NewRating[g][0:5])/5   #opinion on the first item(Average of pairwise items)... change it for more items ***************
          RR[j][k]=NewRating[g][i]   #bekhatere error inha [g][i] ra jabeja nakardam
          k+=1
       j+=1
    
    
    import matplotlib.pyplot as plt
    #%matplotlib inline 
    Iteration = 20
    NR=np.zeros([Iteration, Num_Mem, Num_Item], dtype=object)
    for i in range (Iteration):
      NewRates=np.dot(w, RR)
      NR[i][:][:]=RR
      RR=NewRates
      #print(i, NewRates)
       
    
    # Sorting rates in Descending order
    sort_index = np.argsort(NewRates[0][:]*-1) 
    BestItems_count=20
    BestItems=sort_index[0:BestItems_count]
    #print("BestItems", BestItems) 
    

    RG=BestItems
    
    test_data = NewRating
    
    
    # TP: True Positive
    TP=0
    Teta = 8
    for i in BestItems:
        for u in GroupMemIDs:
           if test_data[u][i]!=0 and test_data[u][i] >= Teta:
              TP+=1
              #print(test_data[u][i])
    SumTP += TP
    
    
    # FP: False positive
    FP=0
    for i in BestItems:
        #print(NewRates[0][i])
        if NewRates[0][i] <= Teta:
           FP+=1
    SumFP += FP
    
    
    # T: Expected recommendations set
    T=0
    for i in range(Num_Movies):
        for u in GroupMemIDs:
           if test_data[u][i]!=0 and test_data[u][i] >= Teta:
              T+=1
    #print("T:", T)
    SumT += T
    
    
    #Evaluation based on Consensus ans Fairness
    # Fairness is defined as the share of group members ui with at least m items in the recommended package for which ui has a high performance.
    
    SatisfiedMembers = 0
    Threshold=0.70
    for u in GroupMemIDs:
        F = 0
        for i in BestItems:
            if NewRating[u][i] >= Threshold:   #bekhatere error jaye [u] va [i] avaz nakardam.
                F += 1
        if F > BestItems_count/2:
            SatisfiedMembers += 1
    SumFairness += SatisfiedMembers /Num_Mem      
       
    #Consensus is a measure of agreement between group members. Here it is pairwise distance between users final opinions.   
    SumC = 0
    for ui in range(Num_Mem):
        for uj in range(Num_Mem):
            SumC += NewRates[ui][0] - NewRates[uj][0]
    SumConsensus += 1 - SumC /  (Num_Mem * Num_Mem)      
    
Precision=SumTP/(SumTP+SumFP)
#print("Precision:", Precision)
Recall=SumTP/SumT
#print("Recall:", Recall)
Fairness = SumFairness/ group
#print ("Fairness:", Fairness) 
Consensus = SumConsensus/ group
#print ("Consensus:", Consensus)    
   

FinalEvaluation[0][0] = Num_Mem
FinalEvaluation[0][1] = Precision
FinalEvaluation[0][2] = Recall
FinalEvaluation[0][3] = Fairness
FinalEvaluation[0][4] = Consensus


print ("personalit scores:", p)
#print ("Num_Mem", Num_Mem)
print ("Precision:", Precision)
#print ("Recall", Recall)
print ("Fairness:", Fairness)
#print ("Consensus", Consensus)
#%%
    
#FinalEvaluation
df1 = pd.DataFrame(FinalEvaluation, columns = ['Group_size', 'Precision', 'Recall', 'Fairness', 'Consensus'])
   
df1.Group_size *= 10
 
plt.scatter('Precision', 'Fairness', 
             s='Group_size',
             alpha=0.5, 
             data=df1)
plt.xlabel("Precision", size=14)
plt.ylabel("Fairness", size=14)
plt.title("bubble sizes indicate Group size", size=14)




