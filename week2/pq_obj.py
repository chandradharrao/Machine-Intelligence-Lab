class MinHeap: #made for A* and Dijkstras Algo
    #node : (dist,node_number)
    
    def __init__(self):
        self.H=[] #heap array
        self.size=0 #size of the heap
        self.indx=-1 #indx of inserted ele
        self.map=dict() #track index of each ele in the heap array

    def get_parent_indx(self,i):
        return (i-1)//2

    def get_left_child_indx(self,i):
        return (2*i)+1

    def get_right_child_indx(self,i):
        return (2*i)+2

    def get_min(self):
        return self.H[0]

    def _lt_(self,el1,el2): #cmp the value(dist) of nodes
        if el1[0]==el2[0]: #if same pty
            #lexi ordering wins ie 'A' wins over 'B'
            return el1[1]<el2[1]

        #if aint same pty,lower value(dist) wins
        return el1[0]<el2[0]

    def siftUp(self,i):
        if self.size==0: return

        v=self.H[i] #cache the child
        j=self.get_parent_indx(i) #grab parent indx

        while i>=1 and self._lt_(v,self.H[j]): #while par exists && child is smaller than par

            self.H[i] = self.H[j] #bring down parent
            self.map[f"{self.H[j][1]}"] = i #update its indx

            i=j #update child's indx
            j=self.get_parent_indx(i) #recalc new par

        #after finding chil'd indx,update 
        self.map[f"{v[1]}"]=i
        self.H[i]=v #place back child
            

    def siftDown(self,i):
        if self.size==0: return
        
        k=i #copy par indx
        v=self.H[k] #par cached
        isHeap=False #assume sub tree !heap

        j=self.get_left_child_indx(i)

        while not isHeap and j<self.size:
            if j+1<self.size and self._lt_(self.H[j+1],self.H[j]): #if right node is smaller than left
                j=j+1
                
            if self._lt_(v,self.H[j]):  #if parent itself is smaller than both childs
                isHeap=True
            else:
                self.H[k]=self.H[j] #move child up
                self.map[f"{self.H[j][1]}"] = k #update child indx

                k=j #move par indx down to shifted chil'd indx
                j=self.get_left_child_indx(k) #recalc left child indx

        self.map[f"{v[1]}"]=k
        self.H[k]=v

    def insert(self,ele):
        self.indx+=1 #update to last inserted loc
        self.H.append(ele)
        self.map[f"{ele[1]}"]=self.indx

        self.size+=1
        self.siftUp(self.indx)

    def deleteMin(self):
        if self.size==0: return

        res=self.H[0]
        self.H[0] = self.H[self.size-1]
        self.map[f"{self.H[0][1]}"]=0
        
        self.H.pop()
        self.map[f"{res[1]}"]=None
        self.size-=1
        self.indx-=1

        self.siftDown(0)
        return res
    
    def decreaseValue(self,ele,newVal):
        if self._lt_(newVal,ele):
            i=self.map[f"{ele[1]}"]
            self.H[i]=newVal
            self.siftUp(i) #since the new ele must be LT existing ele,move up the minPQ
            self.map[f"{ele[1]}"]=None #instead of deleting prev mapping,just make it none
        else: return

    def is_empty(self):
        return self.size==0

    def isPresent_fetch(self,ele):
        #print(f"Ele recieved {ele}")

        #print(f"Map {self.map}")
        i=self.map.get(f"{ele[1]}") #indx of the ele in the heap arr

        if i!=None: #if in heap arr
            ret_ele=self.H[i]
            #print(f"ret_ele {ret_ele}")
            return True,ret_ele
        return False,None

if __name__=="__main__":
    # mH=MinHeap()
    # mH.insert((7,'c'))
    # mH.insert((7,'b'))
    # mH.insert((7,'d'))
    # mH.decreaseValue((7,'d'),(7,'a'))
    # # print(mH.map)
    # print(mH.H)

    # while not mH.is_empty():
    #     print(mH.deleteMin())

    # # print(mH.map)
    # # print(mH.size)

    mH = MinHeap()
    mH.insert((5,1))
    print(mH.H)
    print(mH.deleteMin())

    mH.insert((12,2))
    mH.insert((12,3))
    mH.insert((12,5))
    print(mH.H)
    print(mH.deleteMin())
    
    mH.decreaseValue((12,3),(11,3))
    mH.insert((14,6))
    print(mH.H)
    print(mH.deleteMin())

    mH.insert((13,4))
    print(mH.H)
    print(mH.deleteMin())

    mH.decreaseValue((13,4),(12,4))
    mH.insert((13,9))
    print(mH.H)
    print(mH.deleteMin())

    mH.insert((13,7))
    mH.insert((21,8))
    print(mH.H)
    print(mH.deleteMin())