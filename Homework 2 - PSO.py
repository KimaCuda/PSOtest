
# coding: utf-8

# **Computational Intelligence Homework #2**<br>
# Frederik Darwin<br>
# M10601836

# # Particle Swarm Optimization

# **Initializing problem**
# 
# Maximize:
# \begin{equation*}
# f(x,y) = 8 -  \frac{sin^2(sqrt{(x^2+y^2)})} {(1+0.001*(x^2-y^2))^8} \quad-1 \le x \le 2; -1 \le y \le 1; x + y \ge -1
# \end{equation*}

# **Importing Libraries Needed**

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random as rnd
import math
import copy
import time
get_ipython().run_line_magic('matplotlib', 'inline')


# **Defining Global Variable used**

# In[2]:


start_time = time.time()
#creating population
population = 20
iteration = 100
VLimit = 0.5
CrazinessRate = 0.05


# # 1. Initialization
# Below here are the code to create random X and Y coordinate inside the boundary allowed by the problem. This code also checks the condition X+Y >= -1 and re random new number if the condition is not met. This code also create initial Vx, Vy, r1, r2, c1, and c2

# In[3]:


def createrandom():
    random1 = []
    i = 0
    while i < int(population):
        random2 = []
        # x value
        x = float(format(rnd.uniform(-1,2), '.3f'))
        # y value 
        y = float(format(rnd.uniform(-1,1), '.3f'))
    
        # Velocity x (Vx) and Velocity y (Vy) value (-1 to 1)
        Vx = float(format(rnd.uniform(-1,1), '.3f'))
        Vy = float(format(rnd.uniform(-1,1), '.3f'))
        
        #r1 and r2 unformly distributed (0,1)
        r1 = float(format(rnd.uniform(0,1), '.3f'))
        r2 = copy.deepcopy(r1)
        
        #c1 and c2 random for each individual (0-2)
        c1 = float(format(rnd.uniform(0,2), '.3f'))
        c2 = float(format(rnd.uniform(0,2), '.3f'))
    
        # x+y >= -1 check
        if x+y >= -1:
            random2 += [x,y,Vx,Vy,r1,r2,c1,c2]
            random1.append(random2)
            i = i+1
        else:
            i = i
    return random1


# # 2. Fitness Evaluation
# code below provide fitness evaluation for the first iteration (fitnesscheck) and the next iteration after it (secondfitnesscheck). Please mind that secondfitness check is not just checking second iteration, but all the iteration that comes after first iteration.

# In[4]:


def fitnesscheck():
    randomlist = createrandom()
    for i in range(len(randomlist)):
        x = randomlist[i][0]
        y = randomlist[i][1]
        fitness = float(8 - ((math.sin((math.sqrt(float(x)**2+float(y)**2)))**2)/(1+0.001*(float(x)**2-float(y)**2))**8))
        randomlist[i].append(fitness)
    randomlist_df = pd.DataFrame(randomlist)
    randomlist_df["pBest"] = 0
    randomlist_df["gBest"] = 0
    randomlist_df.rename(columns = {0: 'x', 1:'y', 2:'Vx',3:'Vy', 4:'r1',5:'r2',6:'c1',7:'c2', 8:'Value'},inplace=True)
    return randomlist_df

def secondfitnesscheck(df_next, xyarray):
    fitness_list = []
    for i in range(len(xyarray)):
        x = xyarray[i,0]
        y = xyarray[i,1]
        fitness = float(8 - ((math.sin((math.sqrt(float(x)**2+float(y)**2)))**2)/(1+0.001*(float(x)**2-float(y)**2))**8))
        fitness_list.append(fitness)
    df_next["Value"] = fitness_list
    return df_next


# # 3. pBest and gBest updating
# The code below do calculation to new pBest and gBest for the first iteration after randoming X, Y, r1, r2, c1, and c2 parameter

# In[5]:


def UpdatepBest(randomlist):
    for i in range(len(randomlist)):
        if randomlist['Value'].loc[i] > randomlist['pBest'].loc[i]:
            temp = randomlist["Value"].loc[i]
            randomlist['pBest'].loc[i] = temp
    return randomlist

def UpdategBest(randomlist):
    temp = max(randomlist["Value"])
    if max(randomlist["gBest"]) < temp:
        randomlist["gBest"] = temp
    return randomlist


# **Velocity update**
# \begin{equation*}
# \vec{V_{i}}\left ( t+1 \right ) = w\:  * \: \vec{Vi}\left ( t \right ) + r_{1}c_{1}\left ( \vec{X_{pBest}} - X_{i}\left ( t \right ) \right )+r_{2}c_{2}\left ( \vec{X_{gBest}} - X_{i}\left ( t \right ) \right )
# \end{equation*}
# 
# **Location Update**
# \begin{equation*}
# \vec{X_{i}}\left ( t+1 \right ) = \vec{X_{i}}\left ( t \right )+\vec{V_{i}}\left ( t+1 \right )
# \end{equation*}

# ## **Defining Rule Used** <br>
# **w (Inertia Weight)**<br>
# will have varying value from 1 to 0 as the course move<br>
# \begin{equation*}
# w\left ( t \right ) = \bar{w}-\frac{t}{T} \left ( \bar{w}- \underline{w}\right )
# \end{equation*}
# 
# **c1 and c2**<br>
# c1 and c2 representing each individual in population that it have different psychological behavior, so i will randomly select c1(cognitive behavior) and c2(social behavior) for each individual
# 
# **r1 and r2**<br>
# is a uniformly distributed value between 0-1, i will make it random for each individual.
# 
# **maintaining diversity**<br>
# i will try to create craziness of the particle to maintain diversity

# In[6]:


def inertia():
    w = []
    for i in range(iteration):
        #iterationnum = list(np.arange)
        temp_w = 1 - (i/iteration)*1
        w.append(temp_w)
    return w


# # 4. Velocity and Location Update
# 2 Codes below provide the new velocity and location update after iteration

# In[7]:


#defining velocity update
def velocityupdate(df2, w, xyarray, VxVyarray, r1r2c1c2, pBestLoc_array, gBestLoc):
    Vt = []
    Vt_list = []
    for i in range(len(xyarray)):
        
        Vt.append(w * VxVyarray[i] + r1r2c1c2[i][0]*r1r2c1c2[i][2]*(pBestLoc_array[i] - xyarray[i]) + r1r2c1c2[i][1]*r1r2c1c2[i][3]*(gBestLoc - xyarray[i]))
    Vt = np.array(Vt)
    for i in range(len(Vt)):
        Vt_list.append([Vt[i,0,0],Vt[i,0,1]])
    return Vt_list


# In[8]:


#velocity update
def LocUpdate(df2, w, xyarray, VxVyarray, r1r2c1c2, pBestLoc_array, gBestLoc):
    Vt = velocityupdate(df2, w, xyarray, VxVyarray, r1r2c1c2, pBestLoc_array, gBestLoc)
    VtLimited = VelocityLimit(np.array(Vt))
    VtCrazy = IntroduceCraziness(VtLimited)
    Xt = BoundaryAdhere(xyarray,VtCrazy)
    return VtCrazy, Xt


# # 5. Craziness Parameter
# This code introduce craziness parameter to the flock after updating the velocity, before adding to the location update

# In[9]:


def IntroduceCraziness(VtLimited):
    for i in range(len(VtLimited)):
        for j in range(len(VtLimited[0])):
            a = rnd.random()
            if a < CrazinessRate:
                VtLimited[i][j] = a * VLimit
    return VtLimited


# # 6. Constraint Handling
# This two code below is the code used to handle Velocity Damping and the Boundary Violation by adhering to the boundary. Vlimit can be easily changed from the global variable

# In[10]:


def VelocityLimit(Vt):
    for i in range(len(Vt)):
        for j in range(len(Vt[0])):
            if Vt[i,j]> VLimit:
                Vt[i,j] = VLimit
    return Vt


# In[11]:


def BoundaryAdhere(xyarray,VtLimited):
    Xt = xyarray + VtLimited
    for i in range(len(Xt)):
        if Xt[i,0] < -1:
            Xt[i,0] = -1
        elif Xt[i,0] > 2:
            Xt[i,0] = 2
    for i in range(len(Xt)):
        if Xt[i,1] < -1:
            Xt[i,1] = -1
        elif Xt[i,1] > 1:
            Xt[i,1] = 1
    return Xt


# ## Main code
# This is where all the function created on the top will concatenate and create iterations for the flock. X and Y coordinate for each iteration per individual will be recorded and returned from this function to create the graph model 

# In[12]:


def sec_iteration(df):
    weight = inertia()
    x_list = []
    y_list = []
    df2 = copy.deepcopy(df)
    gBest = max(df["gBest"])
    gBestLoc = np.array(df.loc[df['pBest'] == max(df['pBest']), ['x','y']].values.tolist())
    pBest = []
    pBestLoc = []
    for i in range(len(df)):
        pBest.append(df.iloc[i][9].tolist())
        pBestLoc.append(df.iloc[i][0:2].tolist())
    pBestLoc_array = np.array(pBestLoc)
    for i in range(iteration-1):
        #creating x, y array
        xy = []
        VxVy = []
        r1r2c1c2 = []
        for i in range(len(df)):
            xy.append(df2.iloc[i][0:2].tolist())
            VxVy.append(df2.iloc[i][2:4].tolist())
            r1r2c1c2.append(df2.iloc[i][4:8].tolist())
        xyarray = np.array(xy)
        VxVyarray = np.array(VxVy)
        VtLimited, Xt = LocUpdate(df2, weight[i+1], xyarray, VxVyarray, r1r2c1c2, pBestLoc_array, gBestLoc)
        df_next = copy.deepcopy(df2)
        df_next['x'] = Xt[:,0]
        df_next['y'] = Xt[:,1]
        df_next['Vx'] = VtLimited[:,0]
        df_next['Vy'] = VtLimited[:,1]
        df_next2 = secondfitnesscheck(df_next, Xt)
        df_next3 = UpdategBest(UpdatepBest(df_next2))
        for i in range(len(df_next3)):
            for j in range(len(pBestLoc_array[0])):
                if pBest[i] < df_next3.iloc[i]['pBest']:
                    pBest[i] = df_next3.iloc[i]['pBest']
                    pBestLoc_array[i,j] = df_next3.iloc[i][j]
        if max(df_next3['pBest']) > gBest:
            gBestLoc = np.array(df_next3.loc[df_next3['pBest'] == max(df_next3['pBest']), ['x','y']].values.tolist())
            gBest = max(df_next3['pBest'])
        x_list.append(df_next['x'].tolist())
        y_list.append(df_next['y'].tolist())
        df2 = copy.deepcopy(df_next3)
    return x_list, y_list, df2, VtLimited


# #### PS: This warning can be ignored

# In[13]:


first_iteration = UpdategBest(UpdatepBest(fitnesscheck()))


# In[14]:


#first_iteration


# In[15]:


x,y,z,w= sec_iteration(first_iteration)


# In[16]:


x_first = [[]]
x_first[0] = first_iteration.loc[:,'x'].values.tolist()


# In[17]:


y_first = [[]]
y_first[0] = first_iteration.loc[:,'y'].values.tolist()


# In[18]:


x_all = x_first + x
y_all = y_first + y


# In[19]:


#z


# In[20]:


x_T = list(map(list, zip(*x_all)))


# In[21]:


y_T = list(map(list, zip(*y_all)))


# In[22]:


#x_T


# In[23]:


#y_T


# In[28]:


max(z['pBest'])


# In[24]:


np.mean(z['Value'])


# In[25]:


print("--- %s seconds ---" % (time.time() - start_time))


# # Main Plotting function

# In[26]:


fig = plt.figure(figsize = (15, 15))
ax1 = plt.subplot(212)
for i in range(len(x_T)):
    ax1.scatter(x_T[i], y_T[i])
plt.plot(0.0,0.0, 'ko')
plt.axhline(y=-1, color='r', linestyle='-')
plt.axhline(y=1, color='r', linestyle='-')
plt.axvline(x=-1, color='r', linestyle='-')
plt.axvline(x=2, color='r', linestyle='-')
plt.annotate('global optima', xy=(0, 0), xytext=(-0.9, 0.75),
            arrowprops=dict(facecolor='black'), size = 20)
ax2 = plt.subplot(221)
for i in range(len(x_T)):
    ax2.scatter(x_T[i][0], y_T[i][0])
plt.plot(0.0,0.0, 'ko')
plt.axhline(y=-1, color='r', linestyle='-')
plt.axhline(y=1, color='r', linestyle='-')
plt.axvline(x=-1, color='r', linestyle='-')
plt.axvline(x=2, color='r', linestyle='-')
plt.title('first iteration')
ax3 = plt.subplot(222)
for i in range(len(x_T)):
    ax3.scatter(x_T[i][-1], y_T[i][-1])
plt.plot(0.0,0.0, 'ko')
plt.axhline(y=-1, color='r', linestyle='-')
plt.axhline(y=1, color='r', linestyle='-')
plt.axvline(x=-1, color='r', linestyle='-')
plt.axvline(x=2, color='r', linestyle='-')
plt.title('last iteration')
plt.annotate('global optima', xy=(0, 0), xytext=(1, 0.75),
            arrowprops=dict(facecolor='black'), size = 15)


# In[27]:


print("--- %s seconds ---" % (time.time() - start_time))

