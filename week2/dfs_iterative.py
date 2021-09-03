def gen_path(parent,goal,path):
    print(f"Goal {goal}")
    print(f"Parent {parent}")
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
    path=[]
    n=len(cost)
    parent = [-1]*n
    visited = [False]*n
    stk = []

    stk.append(start_point)

    while len(stk):
        top = stk.pop()
        print(f"Top {top}")

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
                print(f"pushed {nei}")
                stk.append(nei)
                print(f"Stack {stk}")
                parent[nei]=top
    return []

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
    goals = [6, 7, 10]
    print(DFS_Traversal(cost,start, goals))