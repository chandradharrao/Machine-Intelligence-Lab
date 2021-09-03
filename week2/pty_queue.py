class MinHeap:
    def __init__(self):
        self.H=[]
        self.size=0
        self.indx=-1 #indx of inserted ele

    def get_parent_indx(self,i):
        return (i-1)//2

    def get_left_child_indx(self,i):
        return 2*i+1

    def get_right_child_indx(self,i):
        return 2*i+2

    def get_min(self):
        return self.H[0]

    def siftUp(self,i):
        #print(f"i {i}")

        v=self.H[i] #cache the node
        #print(f"v {v}")

        j=self.get_parent_indx(i) #grab parent indx
        #print(f"j {j}")

        while i>=1 and v<self.H[j]: #while there is a upper parent larger than child

            self.H[i] = self.H[j]
            #print(f"Demoted parent : {self.H[i]}")

            i=j
            #print(f"New i {i}")
            j=self.get_parent_indx(i)
            #print(f"New j {j}")

        self.H[i]=v
        #print(f"Intermediate H {self.H}")
            

    def siftDown(self,i):
        print("Called sift down")
        k=i
        v=self.H[k] #ele cached
        isHeap=False
        j=self.get_left_child_indx(i)

        while not isHeap and j<self.size:
            if j+1<self.size and self.H[j+1]<self.H[j]: #if right node is smaller than left
                j=j+1
            if v<self.H[j]:  #if parent itself is smaller than both
                isHeap=True
            else:
                self.H[k]=self.H[j]
                k=j
        self.H[k]=v

    def insert(self,ele):
        #print("*************")
        #print(f"To insert {ele}")
        self.indx+=1
        self.H.append(ele)

        self.size+=1
        self.siftUp(self.indx)
        #print("******************")

    def deleteMin(self):
        res=self.H[0]
        self.H[0] = self.H[self.size-1]
        self.H.pop()
        self.size-=1
        self.indx-=1

        print(f"New H {self.H}")
        self.siftDown(0)
        return res
    
    def decreaseValue(self,i,newVal):
        self.H[i]=newVal
        self.siftUp(i)

    def is_empty(self):
        return self.size==0


mH = MinHeap()
mH.insert(45)
mH.insert(20)
mH.insert(14)
mH.insert(12)
mH.insert(31)
mH.insert(7)
mH.insert(11)
mH.insert(13)
mH.insert(7)
print(mH.H)

sorted=[]
while not mH.is_empty():
    x=mH.deleteMin()
    print(x)
    sorted.append(x)
print(sorted)
