#This weeks code focuses on understanding basic functions of pandas and numpy 
#This will help you complete other lab experiments


# Do not change the function definations or the parameters
import numpy as np
from numpy.core.fromnumeric import mean, shape
import pandas as pd

#input: tuple (x,y)    x,y:int 
def create_numpy_ones_array(shape):
    #return a numpy array with one at all index
    array=None
    #TODO
    array = np.ones(shape)
    return array

#input: tuple (x,y)    x,y:int 
def create_numpy_zeros_array(shape):
    #return a numpy array with zeros at all index
    array=None
    #TODO
    array = np.zeros(shape)
    return array

#input: int  
def create_identity_numpy_array(order):
    #return a identity numpy array of the defined order
    array=None
    #TODO
    array = np.identity(order)
    return array

#input: numpy array
'''def matrix_cofactor(array):
    #return cofactor matrix of the given array
    #array=None
    #TODO
    #print(array)
    #array = np.transpose(np.linalg.det(array)*np.linalg.inv(array))
    return array'''

def minor(i,j,array):
        #print("Input array")
        #print(array)
        cut_array = np.delete(array,i,0)
        #print("After deleteing row {x},array is \n{y}".format(x=i,y=cut_array))
        #print("Dim {x}".format(x=cut_array.shape))
        cut_array = np.delete(cut_array,j,1)
        #print("After deleteing col {x},array is \n{y}".format(x=j,y=cut_array))
        #print("Dim {x}".format(x=cut_array.shape))
        return np.linalg.det(cut_array)

#input: numpy array
def matrix_cofactor(array):

    # return cofactor matrix of the given array

    # TODO

    (m, n) = array.shape  # m,n

    # cofactor matrix

    cofac = np.zeros((m, n))
    original=array
    for i in range(m):
        for j in range(n):
            minor_ele = minor(i, j, array)

            # if even
        
            if (i + j) % 2 == 0:
                minor_val = minor_ele
            else:
                minor_val = -minor_ele
                #print("Minor val = {x}".format(x=minor_val))
           # assert array.all() == original.all()
            cofac[i][j] = minor_val
            #print(cofac)
    array = cofac
    return array

#Input: (numpy array, int ,numpy array, int , int , int , int , tuple,tuple)
#tuple (x,y)    x,y:int 
def f1(X1,coef1,X2,coef2,seed1,seed2,seed3,shape1,shape2):
    #note: shape is of the forst (x1,x2)
    #return W1 x (X1 ** coef1) + W2 x (X2 ** coef2) +b
    # where W1 is random matrix of shape shape1 with seed1
    # where W2 is random matrix of shape shape2 with seed2
    # where B is a random matrix of comaptible shape with seed3
    # if dimension mismatch occur return -1
    ans=None
    #TODO

    try:
        np.random.seed(seed1);
        #print(f"Shape1 type {type(shape1)} & shape {shape1}")
        W1=np.random.rand(*shape1 if type(shape1)=='tuple' else shape1)
        #print(f"W1 {W1}")

        np.random.seed(seed2)
        #print(f"Shape2 type {type(shape2)} & shape{shape2}")
        W2=np.random.rand(*shape2 if type(shape2)=='tuple' else shape2)
        #print(f"W2 {W2}")

        inner_val1=np.power(X1,coef1)
        #print(f"inner val1 shape {inner_val1.shape}")
        inner_val2=np.power(X2,coef2)
        #print(f"inner val2 shape {inner_val2.shape}")
        #print(f"Inner vals {inner_val1},{inner_val1}")

        #ans_partial=(W1*inner_val1)+(W2*inner_val2)
        ans_partial=np.matmul(W1,inner_val1)+np.matmul(W2,inner_val2)
        #print(f"Product {ans_partial}")

        np.random.seed(seed3)
        shape3=ans_partial.shape
        b=np.random.rand(*shape3 if type(shape3)=='tuple' else shape3)

        ans=ans_partial+b
    except Exception as e:
        #print(f"Err {e}")
        ans=-1
    return ans

def fill_with_mode(filename, column):
    """
    Fill the missing values(NaN) in a column with the mode of that column
    Args:
        filename: Name of the CSV file.
        column: Name of the column to fill
    Returns:
        df: Pandas DataFrame object.
        (Representing entire data and where 'column' does not contain NaN values)
        (Filled with above mentioned rules)
    """
    df=None
    df = pd.read_csv(filename)
    to_fill_val = df[column].mode()[0]
    #print(to_fill_val)
    df[column] = df[column].fillna(to_fill_val)

    #check for number of nans
    count = sum(df[column].isnull())
    
    if count == 0:
        #print("Success")
        pass
    else:
        #print("Failure")
        pass
    return df

def fill_with_group_average(df, group, column):
    """
    Fill the missing values(NaN) in column with the mean value of the 
    group the row belongs to.
    The rows are grouped based on the values of another column

    Args:
        df: A pandas DataFrame object representing the data.
        group: The column to group the rows with
        column: Name of the column to fill
    Returns:
        df: Pandas DataFrame object.
        (Representing entire data and where 'column' does not contain NaN values)
        (Filled with above mentioned rules)
    """
    #df=None
    to_fill=df.groupby([group])[column].transform(mean)
    #print(to_fill)
    df[column] = df[column].fillna(to_fill)
    return df


def get_rows_greater_than_avg(df, column):
    """
    Return all the rows(with all columns) where the value in a certain 'column'
    is greater than the average value of that column.

    row where row.column > mean(data.column)

    Args:
        df: A pandas DataFrame object representing the data.
        column: Name of the column to fill
    Returns:
        df: Pandas DataFrame object.
    """
    #df=None
    toCmp = df[column].mean()
    df = df.loc[df[column]>toCmp]
    #print(df)
    return df

