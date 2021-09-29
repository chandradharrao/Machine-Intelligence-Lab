import numpy as np
from pprint import pprint
from numpy.core.numeric import full_like

from numpy.core.shape_base import block

class Tensor:

    def __init__(self, arr, requires_grad=True):

        self.arr = arr
        self.requires_grad = requires_grad

        # When node is created without predecessor the op is denoted as 'leaf'
        # 'leaf' signifies leaf node
        self.history = ['leaf', None, None]
        # History stores the information of the operation that created the Tensor.
        # Check set_history

        # Gradient of the tensor
        self.zero_grad() #set gradient of the tensor to zero
        self.shape = self.arr.shape

    def zero_grad(self):
        self.grad = np.zeros_like(self.arr)

    def set_history(self, op, operand1, operand2):
        self.history = []
        self.history.append(op)
        self.requires_grad = False
        self.history.append(operand1)
        self.history.append(operand2)

        if operand1.requires_grad or operand2.requires_grad:
            self.requires_grad = True

    def __add__(self, other):
        if isinstance(other, self.__class__):
            if self.shape != other.shape:
                raise ArithmeticError(
                    f"Shape mismatch for +: '{self.shape}' and '{other.shape}' ")
            out = self.arr + other.arr
            out_tensor = Tensor(out)
            out_tensor.set_history('add', self, other)

        else:
            raise TypeError(
                f"unsupported operand type(s) for +: '{self.__class__}' and '{type(other)}'")

        return out_tensor


    def __matmul__(self, other):
        if not isinstance(other, self.__class__):
            raise TypeError(
                f"unsupported operand type(s) for matmul: '{self.__class__}' and '{type(other)}'")
        if self.shape[-1] != other.shape[-2]:
            raise ArithmeticError(
                f"Shape mismatch for matmul: '{self.shape}' and '{other.shape}' ")
        out = self.arr @ other.arr
        out_tensor = Tensor(out)
        out_tensor.set_history('matmul', self, other)

        return out_tensor

    def grad_add(self, gradients=None):
        if gradients==None:
            #when backwards call hasnt been given a gradient,then its the call on the loss function L
            #print(bcolors.FAIL + "add_grad recieved null gradients" + bcolors.ENDC)
            gradients=np.ones_like(self.arr)
        
        try:
            #print(bcolors.UNDERLINE + "Trying to compare" + bcolors.ENDC)
            #print(bcolors.OKCYAN+"-------"+bcolors.ENDC)
            #pprint(self.history[1].arr)
            #pprint(self.history[2].arr)
            #print(bcolors.OKCYAN+"-------"+bcolors.ENDC)
            np.testing.assert_array_almost_equal(self.history[1].arr,self.history[2].arr)

            #if successfully tested
            #print(bcolors.HEADER + "Recived same operands" + bcolors.ENDC)

            #d/da (a+a) = 2 is the local gradient
            #incoming gradient = gradients
            #outgoing gradient = local_grad*inc_grad
            op1_grad=np.full_like(self.history[1],2.0)*gradients
            op2_grad = np.full_like(self.history[2],2.0)*gradients
            #print(bcolors.OKCYAN + "The grads of same input are " + bcolors.ENDC)
            #pprint(op1_grad)
            #pprint(op2_grad)
            return (op1_grad,op2_grad)
        except:
            #arrays not equal,then continue as is
            pass
            
        op1_grad = np.ones_like(self.history[1])*gradients
        op2_grad = np.ones_like(self.history[2])*gradients
        return (op1_grad,op2_grad)


        

    def grad_matmul(self, gradients=None):
        # TODO
        if gradients==None:
            gradients=np.ones_like(self.history[1])
        #local_grad*incoming_grad
        op1_grad = gradients*self.history[2]
        op2_grad = gradients*self.history[1]
        return (op1_grad,op2_grad)

    def backward(self, gradients=None):

        #print("*******BACKWARDS CALL***********")
        #print(f"Operation is {self.history[0]}")
        try:
            #print(f"Store grad? {self.history[1].requires_grad or self.history[2].requires_grad}")
            #print("-----------------")
            #print("Input Tensors")
            #pprint(self.history[1].arr)
            #pprint(self.history[2].arr)
            #print("-----------------")
            pass
        except Exception as e: 
            #print("Leaf node doesnt have history data struct!!")
            #print(bcolors.WARNING + e + bcolors.ENDC)
            pass

        #mathematical operation on the tensors
        operation=self.history[0]

        if self.history[0]!='leaf': #not leaf node
            #req_grad is contagious
            self.requires_grad = self.history[1].requires_grad or self.history[2].requires_grad

            #check operation
            if operation=="mult":
                grad_res = self.grad_matmul(gradients)
            if operation=="add":
                grad_res = self.grad_add(gradients)

            #print("----------------")
            #pprint(bcolors.OKCYAN+"Result of grad operation "+bcolors.ENDC)
            #pprint(grad_res)
            #print("----------------")

            for i,inpt in enumerate([self.history[1],self.history[2]]):
                #pprint("Children:")
                #pprint(inpt.arr)
                #call grad on children
                inpt.backward(grad_res[i])
        else:
            #print("Setting Leaf node grad")
            if self.requires_grad:
                #leaf node,store res
                self.grad = gradients
            else:
                self.zero_grad()
            #print(bcolors.OKCYAN+"leaf node grad is "+bcolors.ENDC)
            #pprint(self.grad)


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


# if __name__ == "__main__":
#     a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]])) 
#     b = Tensor(np.array([[3.0, 2.0], [1.0, 5.0]]), requires_grad=False)
#     sgrad = np.array([[1.0, 1.0], [1.0, 1.0]])
#     sans = a+b

#     sans.backward()
    
#     print(bcolors.OKCYAN+"Resultant grad value:" + bcolors.ENDC)
#     pprint(a.grad)
#     print(bcolors.OKGREEN + "Ground truth" + bcolors.ENDC)
#     pprint(sgrad)

#     res = np.testing.assert_array_almost_equal(a.grad, sgrad, decimal=2)

#     if res:
#         print(bcolors.OKGREEN + "PASSED :)"+bcolors.ENDC)
#     else:
#         print(bcolors.FAIL + "FAILED :("+bcolors.ENDC)

