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
        print("************************")
        print(f"Before popping {open_list.H}")
        a_node = open_list.deleteMin()
        #print(a_node)
        print(f"Popped node is {a_node}")
        #print(f"Number of nodes in open list {len(open_list)}")
        #input()

        close_list.append(a_node[1]) #add to visited queue
        #heapq.heapify(close_list)

        if a_node[1] in goals:
            print(f"Goal node is {a_node[1]}")
            print(f"Parent: {parent}")

            print(f"Mappings {open_list.map}")
            find(parent,a_node[1],path)

            temp=[]
            while not path.empty():
                temp.append(path.get())
            path=temp
            
            return path

        #print(type(a_node),a_node)
        #print(type(cost))
        neis = cost[a_node[1]]
        print(f"Neigs of {a_node[1]} are @ distances->{neis}")

        added_neis=0
        for nei,dist in enumerate(neis):
            if nei in close_list: #if already visited
                print(f"Already visited node {nei}")
                continue

            if dist > 0: #if valid path exists
                print(f"Dist is {dist}")
                temp = ((a_node[0]-heuristic[a_node[1]])+dist+heuristic[nei],nei)
                print(f"temp node is {temp}")

                #see if present in open list
                isPresent,open_node = open_list.isPresent_fetch(temp)
                # print(f"isPresent {isPresent}")
                # print(f"temp node after manipulation is {temp}")
                print(f"Open node {open_node}")

                if isPresent and open_list._lt_(temp,open_node): #if present and better,then update
                    print(f"Before Updating temp {temp}")
                    open_list.decreaseValue(open_node,temp)
                    print(f"Parent of {nei} is {a_node[1]}")
                    parent[nei]=a_node[1] #update parent array
                    print(f"Parent: {parent}")
                #if present and not better,ignore
                    
                if not isPresent:
                    added_neis+=1
                    print(f"Before inserting temp {temp}")
                    open_list.insert(temp) #add it
                    print(f"Parent of {nei} is {a_node[1]}")
                    parent[nei]=a_node[1] #update parent array
                    print(f"Parent: {parent}")
                    #print(f"Added {temp[1]}")
        #print(f"Added {added_neis} neighbours to open list")

    return [] #if no path found


def cas1():
    cost = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #0
        #0,1,2, 3, 4  5 6  7  8  9  10
        [0,0,6,-1,-1,-1,3,-1,-1,-1,-1],#1
        [0,6,0,3,2,-1,-1,-1,-1,-1,-1], #2
        [0,-1,3,0,1,5,-1,-1,-1,-1,-1], #3
        [0,-1,2,1,0,8,-1,-1,-1,-1,-1],#4
        [0,-1,-1,5,8,0,-1,-1,-1,5,5], #5
        [0,3,-1,-1,-1,-1,0,1,7,-1,-1],#6
        [0,-1,-1,-1,-1,-1,1,0,-1,3,-1],#7
        [0,-1,-1,-1,-1,-1,7,-1,0,2,-1], #8
        [0,-1,-1,-1,-1,5,-1,3,2,0,3], #9
        [0,-1,-1,-1,-1,5,-1,-1,-1,3,0] #10
    ]

    for i in cost:
        if len(i) != 11:
            print("Not allowed!!!")
            input()

    heuristic = [10,8,5,7,3,6,5,3,1,0]
    start = 1
    goals = [10]

    res=A_star_Traversal(cost,heuristic,start,goals)
    print(res)
    if res==[1,6,7,9,10]:
        print("PASSED")
    else:
        print("FAIL")

def case2():
    cost=[
        [0,0,0,0,0],
        [0,0,1,1,-1],
        [0,1,0,-1,2],
        [0,1,-1,0,2],
        [0,-1,2,2,0]
    ]

    start=1
    goals=[4]
    heuristic=[0,0,0,0,0,0]
    res=A_star_Traversal(cost,heuristic,start,goals)
    print(res)

def case3():
    cost = [[0, 0, 0, 0, 0, 0, 0], 
             [0, 0, 5, 4,-1,-1,-1],  #a
             [0, 5, 0, 1, 5,-1,-1],  #b
             [0, 4, 1, 0, 8,10,-1],  #c
             [0,-1, 5, 8, 0, 2, 6],  #d
             [0,-1,-1,10, 2, 0, 5],  #e
             [0,-1,-1,-1, 6, 5, 0]]  #z
    heuristic = [0,11, 8, 8, 4, 2, 0]
    start=1
    goals=[6]
    res=A_star_Traversal(cost,heuristic,start,goals)
    print(res)

if __name__ == "__main__":
    case2()
