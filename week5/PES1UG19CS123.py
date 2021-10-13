import numpy as np

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

        #managd contagiousness of required_grad
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
        if gradients is None:
            #when backwards call hasnt been given a gradient,then its the call on the loss function L
            gradients=np.ones_like(self.arr)
            
        op1_grad = gradients
        op2_grad = gradients
        return (op1_grad,op2_grad)

    def grad_matmul(self, gradients=None):
        if gradients is None: #grads are none for e.backward() call only
            gradients=np.ones_like(self.arr)

        op1_grad = np.matmul(gradients,np.transpose(self.history[2].arr))
        op2_grad = np.matmul(np.transpose(self.history[1].arr),gradients)
        return (op1_grad,op2_grad)

    def backward(self, gradients=None): #incoming upstream grads 
        #mathematical operation on the tensors
        operation=self.history[0]

        if operation!='leaf': #not leaf node

            #check operation
            if operation=="matmul":
                grad_res = self.grad_matmul(gradients)
            if operation=="add":
                grad_res = self.grad_add(gradients)

            for i,inpt in enumerate([self.history[1],self.history[2]]):
                #children
                #call grad on children
                inpt.backward(grad_res[i])

        else: #leaf node
            if self.requires_grad:
                #leaf node,store res
                self.grad += gradients
            else:
                self.zero_grad()
