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
dfs_path = []
visited = [0]*(len(cost[0])+1)
dfs_stack = []
def dfs_algo(cost, start_point, goals):
    visited[start_point] = 1
    dfs_path.append(start_point)
    print(start_point)
    print(goals)
    print(start_point in goals)
    counter = 0
    if start_point in goals:
        counter += 1
        return True
    print("counter : {}".format(counter))
    called = 0
    for i in range(1, len(cost[0])):
        if (cost[start_point][i] > 0) and (visited[i] != 1) :
            called += 1
            print("called : {}".format(called))
            catch = dfs_algo(cost, i, goals)
            if catch: return catch

#for i in range(1, len(cost[0])+1):
    #print(i)

#print(len(visited))
dfs_algo(cost, 1, goals)
print(dfs_path)