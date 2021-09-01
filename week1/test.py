import numpy as np

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
        print(f"Shape1 type {type(shape1)} & shape {shape1}")
        W1=np.random.rand(*shape1 if type(shape1)=='tuple' else shape1)
        print(f"W1 {W1}")

        np.random.seed(seed2)
        print(f"Shape2 type {type(shape2)} & shape{shape2}")
        W2=np.random.rand(*shape2 if type(shape2)=='tuple' else shape2)
        print(f"W2 {W2}")

        inner_val1=np.power(X1,coef1)
        print(f"inner val1 shape {inner_val1.shape}")
        inner_val2=np.power(X2,coef2)
        print(f"inner val2 shape {inner_val2.shape}")
        print(f"Inner vals {inner_val1},{inner_val1}")

        #ans_partial=(W1*inner_val1)+(W2*inner_val2)
        ans_partial=np.matmul(W1,inner_val1)+np.matmul(W2,inner_val2)
        print(f"Product {ans_partial}")

        np.random.seed(seed3)
        shape3=ans_partial.shape
        b=np.random.rand(*shape3 if type(shape3)=='tuple' else shape3)

        ans=ans_partial+b
        print(f"Ans {ans}")
    except Exception as e:
        print(f"Err {e}")
        ans=-1
    return ans

	
def minor(i,j,array):
        print("Input array")
        print(array)
        cut_array = np.delete(array,i,0)
        print("After deleteing row {x},array is \n{y}".format(x=i,y=cut_array))
        print("Dim {x}".format(x=cut_array.shape))
        cut_array = np.delete(cut_array,j,1)
        print("After deleteing col {x},array is \n{y}".format(x=j,y=cut_array))
        print("Dim {x}".format(x=cut_array.shape))
        return np.linalg.det(cut_array)

#input: numpy array
'''
resources:
https://semath.info/src/cofactor-matrix.html
https://note.nkmk.me/en/python-numpy-delete
https://www.tutorialspoint.com/numpy/numpy_determinant.htm
https://www.geeksforgeeks.org/minors-and-cofactors-of-determinants/
https://gist.github.com/volf52/2eef7d07414669b64b6029fe8eacfe2c
https://www.vedantu.com/maths/cofactor-of-matrices
Finding with the help of adjoint and inverse
'''
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
            assert array.all() == original.all()
            cofac[i][j] = minor_val
            #print(cofac)
    array = cofac
    return array

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
    row_group = df.groupby([group]).mean()
    print(row_group)
    for row in row_group:
        print(f"rows: {row}")
    return df



if __name__ == "__main__":
    res=f1(np.array([[1,2],[3,4]]),3,np.array([[1,2],[3,4]]),2,1,2,3,(3,2),(3,2))

    if(res.all() == np.array([[415.11116764, 604.9332781 ],[187.42695991 ,273.27266349],[112.57538713, 163.6775407 ]]).all()):
        print("PASSED")
    else:
        print("Failed")
    #print(matrix_cofactor(np.array([ [1,2],[1,2] ])))