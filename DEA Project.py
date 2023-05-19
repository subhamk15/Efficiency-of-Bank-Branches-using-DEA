#!/usr/bin/env python
# coding: utf-8

# # <span style='color:Red; font-family:Helvetica; font-size:1em'> Operational Efficiency of Bank Branches using Data Envelopment Analysis </span>
# # <span style='color:Black; background :yellow; font-family:Helvetica; font-size:2em'> Project Report </span>
# # <span style='color:Blue; font-family:Helvetica; font-size:1em'>  Multi-criteria Decision Models </span>

# # <center style='color: Magenta ; font-family:Helvetica; font-size:1em'>Brief Report</center>

# <span style='color: black ; font-family:Helvetica; font-size:1em'> The banking sector has always been a very attractive area for economic research 
# and studies. Calculating the operational efficiency of branches in a bank is 
# important for several reasons:
#     
# 1. Identifying areas for improvement: By measuring the efficiency of 
# different branches, the bank can identify areas where operations could be 
# improved. For example, a branch with low efficiency could be targeted for 
# process improvements or staff training.
#     
# 2. Optimizing resource allocation: Understanding which branches are 
# performing well and which are not can help the bank allocate resources 
# more effectively. For instance, if a branch is operating efficiently, the bank 
# may choose to invest more in marketing or other growth initiatives for that 
# branch.
#     
# 3. Benchmarking: Comparing the performance of different branches against 
# each other and against industry benchmarks can provide valuable insights 
# into best practices and areas where the bank can improve.
#     
# In this project we explore the branches of Central Bank of India by calculating
# its efficiency. We compare efficiency and productivity growth differentials in 
# order to understand which branches were better performers. As a method of study,
# we adopt Data Envelopment Analysis (DEA) approach. DEA models are 
# extensively used for analysing banking firms. Their advantage is the ability to 
# incorporate multiple inputs and multiple outputs as well as various forms of 
# constraints. DEA has been used extensively in studies of the banking industry in 
# developed and developing economies; for individual countries as well as crosscountry comparisons (Aly et al., 1990; Chen, 1998; Sathye, 2001; Casu & 
# Girardone, 2006; Moffat & Valadkhani, 2011).
# Though this framework does not require profit maximization assumption and 
# does not explicitly account for the market structure, it proved to give meaningful 
# results and to serve its purpose to give the estimation of ‘who does better'
#     
# 
# </span>

# # <center style='color: Aqua ; font-family:Helvetica; font-size:1em'>Inputs and Outputs</center>

# <span style='color: black ; font-family:Helvetica; font-size:1em'> 
#                                   
#                            Input Factors
# 1. Operating Staffs: - Operating staffs in a bank are responsible for carrying 
# out the day-to-day operations of the bank. They are the backbone of the 
# bank and play a critical role in ensuring the smooth functioning of the 
# bank's operations. They are the face of the bank for most customers and are 
# responsible for building and maintaining strong relationships with 
# customers.
# 2. Operating Cost: - It includes the overall cost to operate a particular branch 
# and includes costs like rent.
# 3. Total Number of ATMs, Kiosks, Passbook Printers: - ATMs and other 
# electronic machines are used for making deposits and bank spend a large 
# amount of resource in order to operate them.
# 4. Premises Area: - Premises data is used as an input because it is a measure 
# of the physical size of the branch and can provide insights into its capacity 
# to serve customers and carry out transactions. Using floor space as an input 
# in DEA can provide insights into the efficiency of a bank branch and help 
# identify areas for improvement.
#     
#                         Output Factors
# 1. Total number of new Accounts Opened (Savings and Current): - A
# branch that is able to open a large number of new accounts relative to its 
# inputs may be considered more efficient than a branch that is only able to 
# open a small number of accounts with the same inputs.
# 2. Total Amount deposited: - It is the total sum of amount deposited in a 
# given branch for the financial year 2022-2023.
# 
#     
# 
# </span>

# # Data Importing

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


df = pd.read_csv(r'C:\Users\Subham PC\OneDrive\Desktop\Operation Research\SEM 4\Project\Bank1.csv')
df.head(22)


# # Descriptive Statistics

# In[5]:


df.describe()


# # Co-relation Matrix

# In[6]:


plt.figure(figsize = (20,10))
plt.title('Bank)')
corr = df.corr()
sns.heatmap(corr,xticklabels=corr.columns,yticklabels=corr.columns,linewidths=1,annot=True,cmap='seismic')
plt.show()


# # Data Visualisation

# In[7]:


fig = plt.subplots(figsize = (25,5))
sns.barplot(x = df['DMUs'], y= df['Operating Staffs'])


# In[8]:


fig = plt.subplots(figsize = (25,5))
sns.barplot(x = df['DMUs'], y= df['Operating Cost in Lakhs(per year)'])


# In[9]:


fig = plt.subplots(figsize = (25,5))
sns.barplot(x = df['DMUs'], y= df['Premises in thousands m^2'])


# In[10]:


fig = plt.subplots(figsize = (25,5))
sns.barplot(x = df['DMUs'], y= df['Total number of new Accounts Opened (Savings and Current)'])


# In[11]:


fig = plt.subplots(figsize = (25,5))
sns.barplot(x = df['DMUs'], y= df['Total Amount deposited in cr'])


# # Building the model using Gurobi

# In[16]:


get_ipython().run_line_magic('pip', 'install gurobipy')


# In[12]:


import pandas as pd
from itertools import product

import gurobipy as gp
from gurobipy import GRB


# In[14]:


def solve_DEA(target, verbose=True):
    # input-output values for the branches
    inattr = ['Operating Staffs', 'No. of ATMs', 'Operating Cost in Lakhs(per year)', 'Premises in thousands m^2']
    outattr = ['Total number of new Accounts Opened (Savings and Current)', 'Total Amount deposited in cr']
    
    dmus, inputs, outputs = gp.multidict({
        'Adalaj': [{'Operating Staffs': 6, 'No. of ATMs': 8,'Operating Cost in Lakhs(per year)': 5.82000, 'Premises in thousands m^2': 1.000}, {'Total number of new Accounts Opened (Savings and Current)': 297, 'Total Amount deposited in cr': 1.572}],
        'Bhalej': [{'Operating Staffs': 3,'No. of ATMs': 7, 'Operating Cost in Lakhs(per year)': 0.14400, 'Premises in thousands m^2': 1.000}, {'Total number of new Accounts Opened (Savings and Current)': 307, 'Total Amount deposited in cr': 0.545}],
        'Boriavi': [{'Operating Staffs': 4,'No. of ATMs': 8, 'Operating Cost in Lakhs(per year)': 0.90000, 'Premises in thousands m^2': 0.495}, {'Total number of new Accounts Opened (Savings and Current)': 510, 'Total Amount deposited in cr': 0.709}],
        'Borsad': [{'Operating Staffs': 7,'No. of ATMs': 4, 'Operating Cost in Lakhs(per year)': 5.88000, 'Premises in thousands m^2': 1.480}, {'Total number of new Accounts Opened (Savings and Current)': 463, 'Total Amount deposited in cr': 1.241}],
        'Davol': [{'Operating Staffs': 3,'No. of ATMs': 4, 'Operating Cost in Lakhs(per year)': 0.96000, 'Premises in thousands m^2': 1.500}, {'Total number of new Accounts Opened (Savings and Current)': 290, 'Total Amount deposited in cr': 0.739}],
        'Dharmaj': [{'Operating Staffs': 4,'No. of ATMs': 7, 'Operating Cost in Lakhs(per year)': 1.20000, 'Premises in thousands m^2': 0.243}, {'Total number of new Accounts Opened (Savings and Current)': 117, 'Total Amount deposited in cr': 1.133}],
        'Himatnagar': [{'Operating Staffs': 7,'No. of ATMs': 8, 'Operating Cost in Lakhs(per year)': 8.10000, 'Premises in thousands m^2': 1.200}, {'Total number of new Accounts Opened (Savings and Current)': 124, 'Total Amount deposited in cr': 0.118}],
        'Idar': [{'Operating Staffs': 6,'No. of ATMs': 8, 'Operating Cost in Lakhs(per year)': 2.64000, 'Premises in thousands m^2': 1.500}, {'Total number of new Accounts Opened (Savings and Current)': 301, 'Total Amount deposited in cr': 0.795}],
        'Karamsad': [{'Operating Staffs': 3,'No. of ATMs': 8, 'Operating Cost in Lakhs(per year)': 4.56000, 'Premises in thousands m^2': 0.800}, {'Total number of new Accounts Opened (Savings and Current)': 222, 'Total Amount deposited in cr': 0.431}],
        'Kasor': [{'Operating Staffs': 4,'No. of ATMs': 4, 'Operating Cost in Lakhs(per year)': 0.72000, 'Premises in thousands m^2': 0.600}, {'Total number of new Accounts Opened (Savings and Current)': 306, 'Total Amount deposited in cr': 0.364}],
        'Mehsana': [{'Operating Staffs': 6 ,'No. of ATMs': 7, 'Operating Cost in Lakhs(per year)': 9.60000, 'Premises in thousands m^2': 2.600}, {'Total number of new Accounts Opened (Savings and Current)': 156, 'Total Amount deposited in cr': 0.372}],
        'Ode': [{'Operating Staffs': 4,'No. of ATMs': 7, 'Operating Cost in Lakhs(per year)': 3.12000, 'Premises in thousands m^2': 0.100}, {'Total number of new Accounts Opened (Savings and Current)': 489, 'Total Amount deposited in cr': 0.830}],
        'Petlad': [{'Operating Staffs': 5,'No. of ATMs': 8, 'Operating Cost in Lakhs(per year)': 2.16000, 'Premises in thousands m^2': 1.230}, {'Total number of new Accounts Opened (Savings and Current)': 541, 'Total Amount deposited in cr': 1.018}],
        'Palanpur': [{'Operating Staffs': 6,'No. of ATMs': 8, 'Operating Cost in Lakhs(per year)': 7.20000, 'Premises in thousands m^2': 1.261}, {'Total number of new Accounts Opened (Savings and Current)': 187, 'Total Amount deposited in cr': 0.292}],
        'Sarsa': [{'Operating Staffs': 4,'No. of ATMs': 7, 'Operating Cost in Lakhs(per year)': 0.72000, 'Premises in thousands m^2': 1.000}, {'Total number of new Accounts Opened (Savings and Current)': 305, 'Total Amount deposited in cr': 2.308}],
        'Sidhpur': [{'Operating Staffs': 6,'No. of ATMs': 8, 'Operating Cost in Lakhs(per year)': 3.88800, 'Premises in thousands m^2': 1.100}, {'Total number of new Accounts Opened (Savings and Current)': 442, 'Total Amount deposited in cr': 0.543}],
        'Tarapur': [{'Operating Staffs': 5,'No. of ATMs': 8, 'Operating Cost in Lakhs(per year)': 3.96000, 'Premises in thousands m^2': 1.200}, {'Total number of new Accounts Opened (Savings and Current)': 504, 'Total Amount deposited in cr': 1.810}],
        'Unvarsad': [{'Operating Staffs': 5,'No. of ATMs': 8, 'Operating Cost in Lakhs(per year)': 0.25944, 'Premises in thousands m^2': 0.753}, {'Total number of new Accounts Opened (Savings and Current)': 415, 'Total Amount deposited in cr': 2.178}],
        'Unjha': [{'Operating Staffs': 5,'No. of ATMs': 8, 'Operating Cost in Lakhs(per year)': 7.68000, 'Premises in thousands m^2': 1.392}, {'Total number of new Accounts Opened (Savings and Current)': 261, 'Total Amount deposited in cr': 1.237}],
        'VV Nagar': [{'Operating Staffs': 4,'No. of ATMs': 4, 'Operating Cost in Lakhs(per year)': 3.93936, 'Premises in thousands m^2': 1.641}, {'Total number of new Accounts Opened (Savings and Current)': 128, 'Total Amount deposited in cr': 0.440}],
        'Vasai': [{'Operating Staffs': 3,'No. of ATMs': 7, 'Operating Cost in Lakhs(per year)': 0.60000, 'Premises in thousands m^2': 0.924}, {'Total number of new Accounts Opened (Savings and Current)': 445, 'Total Amount deposited in cr': 0.513}],
        'Vavol': [{'Operating Staffs': 6,'No. of ATMs': 8, 'Operating Cost in Lakhs(per year)': 0.42000, 'Premises in thousands m^2': 0.700}, {'Total number of new Accounts Opened (Savings and Current)': 357, 'Total Amount deposited in cr': 0.981}]
    })
    
    ### Create LP model
    model = gp.Model('DEA')
    
    # Decision variables
    wout = model.addVars(outattr, name="outputWeight")
    win = model.addVars(inattr, name="inputWeight")

    # Constraints
    ratios = model.addConstrs( ( gp.quicksum(outputs[h][r]*wout[r] for r in outattr ) 
                                - gp.quicksum(inputs[h][i]*win[i] for i in inattr ) 
                                <= 0 for h in dmus ), name='ratios' )
    
    normalization = model.addConstr((gp.quicksum(inputs[target][i]*win[i] for i in inattr ) == 1 ),
                                    name='normalization')
    
    # Objective function
    
    model.setObjective( gp.quicksum(outputs[target][r]*wout[r] for r in outattr ), GRB.MAXIMIZE)
    
    # Run optimization engine
    if not verbose:
        model.params.OutputFlag = 0
    model.optimize()
    
    # Print results
    print(f"\nThe efficiency of target DMU {target} is {round(model.objVal,3)}") 
    
    print("__________________________________________________________________")
    print(f"The weights for the inputs are:")
    for i in inattr:
        print(f"For {i}: {round(win[i].x,3)} ") 
        
    print("__________________________________________________________________")
    print(f"The weights for the outputs are")
    for r in outattr:
        print(f"For {r} is: {round(wout[r].x,3)} ") 
    print("__________________________________________________________________\n\n")  
    
    return model.objVal


# In[15]:


dmus = ['Adalaj','Bhalej','Boriavi', 'Borsad', 'Davol','Dharmaj','Himatnagar','Idar','Karamsad','Kasor','Mehsana','Ode','Petlad', 'Palanpur', 'Sarsa', 'Sidhpur', 'Tarapur','Unvarsad', 'Unjha',  'VV Nagar', 'Vasai', 'Vavol']


# # Solving DEA model for each DMU

# In[16]:


performance = {}
for h in dmus:    
    performance[h] = solve_DEA(h, verbose=False)


# # Identifying efficient and inefficient DMUs
# 
# # Sorting branches in descending efficiency number

# In[17]:


sorted_performance = {k: v for k, v in sorted(performance.items(), key=lambda item: item[1], reverse = True)}

efficient = []
inefficient = []

for h in sorted_performance.keys():
    if sorted_performance[h] >= 0.9999999:
        efficient.append(h) 
    if sorted_performance[h] < 0.9999999:
        inefficient.append(h) 
        
print('____________________________________________')
print(f"The efficient DMUs are:")
for eff in efficient:
    print(f"The performance value of DMU {eff} is: {round(performance[eff],3)}") 
    
print('____________________________________________')
print(f"The inefficient DMUs are:")
for ine in inefficient:
    print(f"The performance value of DMU {ine} is: {round(performance[ine],3)}") 


# # Plotting the Results

# In[18]:


import pandas as pd
import numpy as np
import chart_studio.plotly as py
import seaborn as sns
import plotly.express as px
import cufflinks as cf
get_ipython().run_line_magic('matplotlib', 'inline')

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected = True)
cf.go_offline()


# In[19]:


import plotly.graph_objects as go
px.line(df, x = 'DMUs', y = 'Total Amount deposited in cr', labels = {'x' : 'DMUs', 'y' : 'Total Amount deposited in cr'})


# In[20]:


import plotly.graph_objects as go
px.bar(df, x = 'DMUs', y = 'Total Amount deposited in cr', labels = {'x' : 'DMUs', 'y' : 'Total Amount deposited in cr'})


# In[21]:


px.scatter(df, x = 'DMUs', y = 'Total Amount deposited in cr', labels = {'x' : 'DMUs', 'y' : 'Total Amount deposited in cr'})


# In[22]:


px.pie(df, values = 'Total Amount deposited in cr', names = 'DMUs', title = 'Percentage of deposits in different branches', color_discrete_sequence = px.colors.sequential.RdBu)


# In[23]:


px.pie(df, values = 'Total number of new Accounts Opened (Savings and Current)', names = 'DMUs', title = 'Percentage of new accounts opened in different branches', color_discrete_sequence = px.colors.sequential.RdBu)


# In[24]:


fig = px.scatter_3d(df, x = 'DMUs', y = 'Total number of new Accounts Opened (Savings and Current)', z = 'Total Amount deposited in cr')
fig


# In[28]:


df1 = pd.read_csv(r'C:\Users\Subham PC\OneDrive\Desktop\Operation Research\SEM 4\Project\Bank1.csv')
df1.head(20)


# In[30]:


df1['Efficient Operating Staffs'] = df1['Operating Staffs'].apply(np.ceil)
df1


# In[31]:


fig = go.Figure(data=[
    go.Bar(name='Operating Staffs', x= df1['DMUs'], y= df1['Operating Staffs'], text = df1['Operating Staffs'] ),
    go.Bar(name='Efficient Operating Staffs', x=df1['DMUs'], y= df1['Efficient Operating Staffs'], text = df1['Efficient Operating Staffs'])
])
# Change the bar mode
fig.update_layout(barmode='group')
fig.update_traces(texttemplate='%{text:.0s}', textposition='outside')
fig.show()


# In[32]:


fig = go.Figure(data=[
    go.Line(name='Operating Staffs', x= df1['DMUs'], y= df1['Operating Staffs'], text = df1['Operating Staffs'] ),
    go.Line(name='Efficient Operating Staffs', x=df1['DMUs'], y= df1['Efficient Operating Staffs'], text = df1['Efficient Operating Staffs'])
])
# Change the bar mode
fig.update_layout(barmode='group')
fig.show()


# In[36]:


df = pd.read_csv(r'C:\Users\Subham PC\OneDrive\Desktop\Operation Research\SEM 4\Project\graph2.csv')
df.head(22)


# In[37]:


import plotly.graph_objects as go
px.bar(df, x = 'Branch', y = 'Score', labels = {'x' : 'Branch', 'y' : 'Efficiency Score'}, title="Efficiency Score")


# # <center style='color:Brown ; font-family:Helvetica; font-size:1em'>Conclusion</center>

# <span style='color: black ; font-family:Helvetica; font-size:1em'> In this project the efficiency of Central Bank of India under the Gandhinagar region was evaluated. Our analysis focused on the relative evaluation of performance of 
# branches. 
#     
# As a method of study, we used Data Envelopment Analysis (DEA). DEA models 
# have been extensively used for analysing bank branches. Their advantage is the 
# ability to incorporate multiple inputs and multiple outputs as well as various 
# forms of constraints. 
#     
# At first, we used the CCR model and found that 11 branches of the bank were 
# efficient and the other 11 were inefficient. Next, we used the BCC model which 
# gave us 14 efficient branches and 8 inefficient ones. Later, to rank the already 
# efficient branches we used the concept of super efficiency. For super efficiency
# model, we used Anderson and Peterson model on our CCR model.
# Calculating efficiency of branches is an interesting phenomenon for a banking 
# sector to optimise their means of resources. Many inputs are reduced by firms 
# (makes banks more efficient, make them squeeze their resource usage). 
#     
# 
# </span>

# In[ ]:




