import heapq
import queue

'''****HANDLE EDGE CASES LIKE NO PATH,one path etc*****'''

'''*****OPTIMIZE PATH COMPRESSION*****'''
def find(parent,goal_node,res):
    if parent[goal_node]==-1:
        #res.append(goal_node)
        res.put(goal_node)
    else:
        #res.append(goal_node)
        res.put(goal_node)
        find(parent,parent[goal_node],res)

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
    path = queue.LifoQueue()

    open_list = []
    heapq.heapify(open_list)

    close_list = [] #just store node number
    heapq.heapify(close_list)

    open_list.append((0+heuristic[start_point],start_point)) #starting node (f(n),node_#)
    heapq.heapify(open_list)

    parent = [-1]*len(heuristic) #stores the parent number

    while open_list: #not empty

        a_node = heapq.heappop(open_list)

        #print(a_node)
        print("************************")
        print(f"Popped node is {a_node}")
        #print(f"Number of nodes in open list {len(open_list)}")
        #input()

        close_list.append(a_node[1]) #add to visited queue
        #heapq.heapify(close_list)

        if a_node[1] in goals:
            print(f"Goal node is {a_node[1]}")
            print(f"Parent: {parent}")
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
                isPresent=False
                for indx,open_node in enumerate(open_list):
                    if open_node[1]==nei:
                        #if present
                        isPresent=True
                        break
                    
                if isPresent and dist+heuristic[nei]<open_node[0]: #if present and better,then update
                    open_list[indx] = temp
                    print(f"Parent of {nei} is {a_node[1]}")
                    parent[nei]=a_node[1] #update parent
                    print(f"Parent: {parent}")
                    heapq.heapify(open_list)
                #if present and not better,ignore
                    
                if not isPresent:
                    added_neis+=1
                    open_list.append(temp) #add it
                    print(f"Parent of {nei} is {a_node[1]}")
                    parent[nei]=a_node[1] #update parent
                    print(f"Parent: {parent}")
                    heapq.heapify(open_list)
                    #print(f"Added {temp[1]}")
        #print(f"Added {added_neis} neighbours to open list")

    return [] #if no path found


if __name__ == "__main__":
    cost = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 5, 9, -1, 6, -1, -1, -1, -1, -1],
            [0, -1, 0, 3, -1, -1, 9, -1, -1, -1, -1],
            [0, -1, 2, 0, 1, -1, -1, -1, -1, -1, -1],
            [0, 6, -1, -1, 0, -1, -1, 5, 7, -1, -1],
            [0, -1, -1, -1, 2, 0, -1, -1, -1, 2, -1],
            [0, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1],
            [0, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1],
            [0, -1, -1, -1, -1, 2, -1, -1, 0, -1, 8],
            [0, -1, -1, -1, -1, -1, -1, -1, -1, 0, 7],
            [0, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0]]
    heuristic = [0, 5, 7, 3, 4, 6, 0, 0, 6, 5, 0]
    start = 1
    goals = [11]

    print(A_star_Traversal(cost,heuristic,start,goals))