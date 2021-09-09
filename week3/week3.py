'''
Assume df is a pandas dataframe object of the dataset given
'''

from os import name
import numpy as np
import pandas as pd
import random
from pprint import pprint

'''
###### Edge cases ########
like log 0
p or n == 0
same value for information gain what to do?
'''

'''
Entropy of entire dataset = - sum(p*logp) for all i=1 to N where N is num of attributes
'''

'''
ID3 Ago:

compute E(S)
for attribute in all_attributes:
    for caltegorical_value in attribute.values:
        calc entropy of categorical_value
calc the gain and pick attr with highest gain
        
'''

def get_entropy(counts,n):
    entropy=0
    for _,v in counts.items():
        pi=v/n
        entropy+=(pi)*np.log2(pi)
    return entropy


'''Calculate the entropy of the enitre dataset'''
# input:pandas_dataframe
# output:int/float
def get_entropy_of_dataset(df):
    last_col = df.columns.values[-1]
    # print(f"Last col name {last_col}")
    # u_attrs = df[last_col].unique()
    counts = {}
    # print(f"Attrs of last col {u_attrs}")
    n=len(df[last_col])
    # print(f"Total #attrs {n}")
    # for u_attr in u_attrs:
    #     count=0
    #     for _ in df.loc[df[last_col]==u_attr]: 
    #         count+=1
    #     counts[u_attr]=count
    #     print(f"The count of {u_attr} is {counts[u_attr]}")
    counts=df[last_col].value_counts()
    # print(f"Counts of last col attrs {counts}")
        
    
    entropy=-get_entropy(counts,n)
    # print(f"Entropy {entropy}")
    return entropy


'''Return avg_info of the attribute provided as parameter'''
# input:pandas_dataframe,str   {i.e the column name ,ex: Temperature in the Play tennis dataset}
# output:int/float
def get_avg_info_of_attribute(df, attribute):
    last_col = df.columns.values[-1]
    #print(f"Last col name {last_col}")

    n=len(df[last_col])
    #print(f"Total #attrs {n}")

    u_vals = df[attribute].unique()
    #print(f"U_attrs of {attribute} is {u_vals}")

    counts={} # {sunny : {p:7,n:12 }}
    lens=[] # num 7+12=19

    for u_val in u_vals:
        mini_df= df[df[attribute]==u_val]
        #print(f"The minidf of {u_val} is ")
        #print(mini_df)

        counts[u_val]=mini_df[last_col].value_counts()
        #print(f"counts of attr_val {u_val} is {counts[u_val]}")
        lens.append(len(mini_df))
        #print(f"Number of enetries of minidf for attr {attribute} is {n}")

    avg_info=0
    for i,u_val in enumerate(u_vals):
        entropy_val = get_entropy(counts[u_val],lens[i])
        #print(f"Entropy of attr_val {u_val} is {entropy_val}")
        avg_info+=(lens[i]/n)*entropy_val
    avg_info=-avg_info
    return avg_info


'''Return Information Gain of the attribute provided as parameter'''
# input:pandas_dataframe,str
# output:int/float
def get_information_gain(df, attribute):
    information_gain=get_entropy_of_dataset(df)-get_avg_info_of_attribute(df,attribute)
    return information_gain

#input: pandas_dataframe
#output: ({dict},'str')
def get_selected_attribute(df):
    '''
    Return a tuple with the first element as a dictionary which has IG of all columns 
    and the second element as a string with the name of the column selected

    example : ({'A':0.123,'B':0.768,'C':1.23} , 'C')
    '''
    cols = df.columns.values[:-1]
    sel_attr={}
    max_info_gain = {"val":float("-inf"),"name":None}
    for col_name in cols:
        sel_attr[col_name]=get_information_gain(df,col_name)
        if sel_attr[col_name]>max_info_gain['val']:
            max_info_gain["val"]=sel_attr[col_name]
            max_info_gain["name"]=col_name
    #print((sel_attr,max_info_gain['name']))
    return (sel_attr,max_info_gain['name'])


if __name__=="__main__":
    outlook = 'overcast,overcast,overcast,overcast,rainy,rainy,rainy,rainy,rainy,sunny,sunny,sunny,sunny,sunny'.split(
        ',')
    temp = 'hot,cool,mild,hot,mild,cool,cool,mild,mild,hot,hot,mild,cool,mild'.split(
        ',')
    humidity = 'high,normal,high,normal,high,normal,normal,normal,high,high,high,high,normal,normal'.split(
        ',')
    windy = 'FALSE,TRUE,TRUE,FALSE,FALSE,FALSE,TRUE,FALSE,TRUE,FALSE,TRUE,FALSE,FALSE,TRUE'.split(
        ',')
    play = 'yes,yes,yes,yes,yes,yes,no,yes,no,no,no,no,yes,yes'.split(',')
    dataset = {'outlook': outlook, 'temp': temp,
                'humidity': humidity, 'windy': windy, 'play': play}
    df = pd.DataFrame(dataset, columns=[
                        'outlook', 'temp', 'humidity', 'windy', 'play'])

    #(get_avg_info_of_attribute(df, 'outlook'))
    columns = ['outlook', 'temp', 'humidity', 'windy', 'play']
    ans = get_selected_attribute(df)
    dictionary = ans[0]
    flag = (dictionary['outlook'] >= 0.244 and dictionary['outlook'] <= 0.248) and (dictionary['temp'] >= 0.0292 and dictionary['temp'] <= 0.0296) and (
        dictionary['humidity'] >= 0.150 and dictionary['humidity'] <= 0.154) and (dictionary['windy'] >= 0.046 and dictionary['windy'] <= 0.05) and (ans[1] == 'outlook')