def dfs(visited,v,cost,path,goals):
    path.append(v)
    visited[v]=True
    print(f"Visited {v}")

    if v in goals:
        print(f"Goal reached {v}")
        return True #break out of recursion as well as for loop

    neis = cost[v]
    for nei,dist in enumerate(neis):

        if dist>0 and not visited[nei]:
            doBreak = dfs(visited,nei,cost,path,goals)
            if doBreak: return True
        


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
    visited = [False]*len(cost)
    dfs(visited,start_point,cost,path,goals)
    return path

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