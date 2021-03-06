import pq_obj as pq
import queue

'''****HANDLE EDGE CASES LIKE NO PATH,one path etc*****'''
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

    open_list = pq.MinHeap()

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
    if len(goals) ==0: return []
    
    path=[]
    n=len(cost)
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