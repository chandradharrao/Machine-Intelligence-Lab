class MinHeap:
    def __init__(self):
        self.H=[]
        self.size=0
        self.indx=-1 #indx of inserted ele
        self.map=dict() #track index of each ele

    def get_parent_indx(self,i):
        return (i-1)//2

    def get_left_child_indx(self,i):
        return 2*i+1

    def get_right_child_indx(self,i):
        return 2*i+2

    def get_min(self):
        return self.H[0]

    def cmp(self,el1,el2): #cmp the pty of nodes
        if el1[0]==el2[0]: #if same pty
            #lexi
            return el1[1]<el2[1]

        #if aint same pty,normal
        return el1[0]<el2[0]

    def siftUp(self,i):
        v=self.H[i] #cache the node
        j=self.get_parent_indx(i) #grab parent indx

        while i>=1 and self.cmp(v,self.H[j]): #while there is a upper parent larger than child

            self.H[i] = self.H[j]
            self.map[f"{self.H[j][1]}"] = i

            i=j
            j=self.get_parent_indx(i)

        self.map[f"{v[1]}"]=i
        self.H[i]=v
            

    def siftDown(self,i):
        if self.size==0: return
        
        k=i
        v=self.H[k] #ele cached
        isHeap=False
        j=self.get_left_child_indx(i)

        while not isHeap and j<self.size:
            if j+1<self.size and self.cmp(self.H[j+1],self.H[j]): #if right node is smaller than left
                j=j+1
                
            if self.cmp(v,self.H[j]):  #if parent itself is smaller than both
                isHeap=True
            else:
                self.H[k]=self.H[j]
                self.map[f"{self.H[j][1]}"] = k
                k=j
                j=self.get_left_child_indx(k)

        self.map[f"{v[1]}"]=k
        self.H[k]=v

    def insert(self,ele):
        self.indx+=1
        self.H.append(ele)
        self.map[f"{ele[1]}"]=self.indx

        self.size+=1
        self.siftUp(self.indx)

    def deleteMin(self):
        res=self.H[0]
        self.H[0] = self.H[self.size-1]
        self.map[f"{self.H[0][1]}"]=0
        
        self.H.pop()
        self.map.pop(f"{res[1]}")
        self.size-=1
        self.indx-=1

        self.siftDown(0)
        return res
    
    def decreaseValue(self,ele,newVal):
        i=self.map[f"{ele[1]}"]
        self.H[i]=newVal
        self.siftUp(i)
        self.map.pop(f"{ele[1]}")

    def is_empty(self):
        return self.size==0

if __name__=="__main__":
    mH=MinHeap()
    mH.insert((7,'c'))
    mH.insert((7,'b'))
    mH.insert((7,'d'))
    mH.decreaseValue((7,'d'),(7,'a'))
    # print(mH.map)
    print(mH.H)

    while not mH.is_empty():
        print(mH.deleteMin())

    # print(mH.map)
    # print(mH.size)