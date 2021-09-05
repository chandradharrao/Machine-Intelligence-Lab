import queue

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
        #print(f"Map {self.map}")
        self.H[0] = self.H[self.size-1]
        self.map[f"{self.H[0][1]}"]=0
        
        self.H.pop()
        del self.map[f"{res[1]}"]
        self.size-=1
        self.indx-=1

        self.siftDown(0)
        return res
    
    def decreaseValue(self,ele,newVal):
        if self._lt_(newVal,ele):
            i=self.map[f"{ele[1]}"]
            self.H[i]=newVal
            self.siftUp(i) #since the new ele must be LT existing ele,move up the minPQ
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

"""
You can create any other helper funtions.
Do not modify the given functions
"""

'''
cost[i][j] -> distance of moving from ith node to jth node
heurestic[i] -> heuristic estimate of distance from node i to the goal node
'''

def find(parent,goal,res):
    while parent[goal] != -1:
        res.put(goal)
        goal = parent[goal]
    if parent[goal]==-1:
        res.put(goal)

def A_star_Traversal(cost, heuristic, start_point, goals):
    """
    Perform A* Traversal and find the optimal path 
    Args:
        cost: cost matrix (list of floats/int)
        heuristic: heuristics for A* (list of floats/int)
        start_point: Staring node (int)
        goals: Goal states (list of ints)
    Returns:
        path: path to goal state obtained from A*(list of ints)
    """
    n=len(heuristic)

    if start_point not in range(0,n): return []

    if len(goals) ==0: return []
    
    path = queue.LifoQueue()

    open_list = MinHeap()

    close_list = [] #just store node number

    open_list.insert((0+heuristic[start_point],start_point)) #starting node (f(n),node_#)

    parent = [-1]*n #stores the parent number

    while not open_list.is_empty(): #not empty
        #print("************************")
        #print(f"Before popping {open_list.H}")
        a_node = open_list.deleteMin()
        #print(a_node)
        #print(f"Popped node is {a_node}")
        #print(f"Number of nodes in open list {len(open_list)}")
        #input()

        close_list.append(a_node[1]) #add to visited queue
        #heapq.heapify(close_list)

        if a_node[1] in goals:
            #print(f"Goal node is {a_node[1]}")
            #print(f"Parent: {parent}")

            #print(f"Mappings {open_list.map}")
            find(parent,a_node[1],path)

            temp=[]
            while not path.empty():
                temp.append(path.get())
            path=temp
            
            return path

        #print(type(a_node),a_node)
        #print(type(cost))
        neis = cost[a_node[1]]
        #print(f"Neigs of {a_node[1]} are @ distances->{neis}")

        added_neis=0
        for nei,dist in enumerate(neis):
            if nei in close_list: #if already visited
                #print(f"Already visited node {nei}")
                continue

            if dist > 0: #if valid path exists
                #print(f"Dist is {dist}")
                temp = ((a_node[0]-heuristic[a_node[1]])+dist+heuristic[nei],nei)
                #print(f"temp node is {temp}")

                #see if present in open list
                isPresent,open_node = open_list.isPresent_fetch(temp)
                # print(f"isPresent {isPresent}")
                # print(f"temp node after manipulation is {temp}")
                #print(f"Open node {open_node}")
                
                if isPresent and open_list._lt_(temp,open_node): #if present and better,then update
                    #print(f"Before Updating,temp {temp}")
                    open_list.decreaseValue(open_node,temp)
                    #print(f"Parent of {nei} is {a_node[1]}")
                    parent[nei]=a_node[1] #update parent array
                    #print(f"Parent: {parent}")
                #if present and not better,ignore
                    
                if not isPresent:
                    added_neis+=1
                    #print(f"Before inserting temp {temp}")
                    open_list.insert(temp) #add it
                    #print(f"Parent of {nei} is {a_node[1]}")
                    parent[nei]=a_node[1] #update parent array
                    #print(f"Parent: {parent}")
                    #print(f"Added {temp[1]}")
        #print(f"Added {added_neis} neighbours to open list")

    return [] #if no path found

def gen_path(parent,goal,path):
    #print(f"Goal {goal}")
    #print(f"Parent {parent}")
    while parent[goal] != -1:
        path.append(goal)
        goal = parent[goal]
    if parent[goal]==-1:
        path.append(goal)

def DFS_Traversal(cost, start_point, goals):
    """
    Perform DFS Traversal and find the optimal path 
        cost: cost matrix (list of floats/int)
        start_point: Staring node (int)
        goals: Goal states (list of ints)
    Returns:
        path: path to goal state obtained from DFS(list of ints)
    """
    n_goals = len(goals)
    if n_goals ==0: return []
    n=len(cost)
    if n == 0: return []

    if start_point<=0 or start_point>n_goals: return []
    
    path=[]
    parent = [-1]*n
    visited = [False]*n
    stk = []

    stk.append(start_point)

    while len(stk):
        top = stk.pop()
        #print(f"Top {top}")

        if visited[top]:
            continue

        if top in goals:
            gen_path(parent,top,path)
            
            return path[::-1]
            
        visited[top]=True
        neis = cost[top]
        for i in range(n-1,-1,-1):
            nei = i
            dist=neis[i]
            if dist>0 and not visited[nei]:
                #print(f"pushed {nei}")
                stk.append(nei)
                #print(f"Stack {stk}")
                parent[nei]=top
    return []