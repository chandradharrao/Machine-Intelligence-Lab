import sys
import importlib
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--SRN', required=True)

args = parser.parse_args()
subname = args.SRN


try:
   mymodule = importlib.import_module(subname)
except:
    print("Rename your written program as YOUR_SRN.py and run python3.7 SampleTest.py --SRN YOUR_SRN ")
    sys.exit()



def testcase(mymodule):
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
    
    cost1 = [
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
    heuristic1 = [0,10,8,5,7,3,6,5,3,1,0]
    
    cost2 = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 5, 9, -1, 6, -1, -1, -1, -1, -1, -1],
            [0, -1, 0, 3, -1, -1, 9, -1, -1, -1, -1, -1],
            [0, -1, 2, 0, 1, -1, -1, -1, -1, -1, -1, -1],
            [0, 6, -1, -1, 0, -1, -1, 5, 7, -1, -1, -1],
            [0, -1, -1, -1, 2, 0, -1, -1, -1, 2, -1, -1],
            [0, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1],
            [0, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1],
            [0, -1, -1, -1, -1, 2, -1, -1, 0, -1, 8, -1],
            [0, -1, -1, -1, -1, -1, -1, -1, -1, 0, 7, -1],
            [0, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, -1],
            [0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0]]
    heuristic2 = [0, 5, 7, 3, 4, 6, 0, 0, 6, 5, 0, 10]
    
    cost3 = [[0, 0, 0, 0, 0, 0, 0], 
             [0, 0, 5, 4,-1,-1,-1],  #a
             [0, 5, 0, 1, 5,-1,-1],  #b
             [0, 4, 1, 0, 8,10,-1],  #c
             [0,-1, 5, 8, 0, 2, 6],  #d
             [0,-1,-1,10, 2, 0, 5],  #e
             [0,-1,-1,-1, 6, 5, 0]]  #z
    heuristic3 = [0, 11, 8, 8, 4, 2, 0]
    
    cost4 = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
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
    heuristic4 = [0, 5, 7, 3, 4, 6, 0, 0, 6, 5, 0]

    cost5 = [[0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 3, -1, -1, -1, -1, 2],
            [0, -1, 0, 5, 10, -1, -1, -1],
            [0, -1, -1, 0, 2, -1, 1, -1],
            [0, -1, -1, -1, 0, 4, -1, -1],
            [0, -1, -1, -1, -1, 0, -1, -1],
            [0, -1, -1, -1, -1, 3, 0, -1],
            [0, -1, -1, 1, -1, -1, 4, 0]]  # https://www.geeksforgeeks.org/search-algorithms-in-ai/
    heuristic5 = [0, 7, 9, 4, 2, 0, 3, 5]
    
    cost6 = [[0, 0, 0, 0, 0, 0],
        [0, 0, 6, -1, -1, -1],
        [0, 6, 0, 3, 3, -1],
        [0, -1, 3, 0, 1, 7],
        [0, -1, 3, 1, 0, 8],
        [0, -1, -1, 7, 8, 0]]    
    heuristic6 = [0, 10, 8, 7, 7, 3]
    
    cost7 = [
            [0,0,0,0,0],
            [0,0,1,1,-1],
            [0,1,0,-1,2],
            [0,1,-1,0,2],
            [0,-1,2,2,0]]
    heuristic7 = [0,0,0,0,0]
    
    start = 1
    goals = [6, 7, 10]
    
    aStarTestCases = [
        (cost, heuristic, start, goals),
        ([],[],start, goals), # Empty graph,
        (cost, heuristic, 15, goals),
        (cost, heuristic, start, []),
        (cost2, heuristic2, start, [11]),
        (cost2, heuristic2, start, [6, 7, 10, start]),
        (cost, heuristic, start, list(range(start + 1, len(cost)))),
        
        (cost1, heuristic1, 1, [10]),
        (cost7, heuristic7, 1, [4]),
        
        (cost3, heuristic3, start, [5]),
        (cost3, heuristic3, start, [6]),
        
        (cost4, heuristic4, 1, [1] ),
        (cost4, heuristic4, 1, [2] ),
        (cost4, heuristic4, 1, [3] ),
        (cost4, heuristic4, 1, [4] ),
        (cost4, heuristic4, 1, [5] ),
        (cost4, heuristic4, 1, [6] ),
        (cost4, heuristic4, 1, [7] ),
        (cost4, heuristic4, 1, [8] ),
        (cost4, heuristic4, 1, [9] ),
        (cost4, heuristic4, 1, [10] ),
        (cost4, heuristic4, 1, [6, 7, 10] ),
        (cost4, heuristic4, 1, [3, 4, 7, 10] ),
        (cost4, heuristic4, 1, [5, 9, 4] ),
        (cost4, heuristic4, 1, [4, 8, 10] ),
        (cost4, heuristic4, 1, [2, 8, 5] ),
        (cost4, heuristic4, 1, [7, 9, 10] ),
        (cost4, heuristic4, 1, [10, 6, 8, 4] ),
        (cost4, heuristic4, 1, [9, 7, 5, 10] ),
        (cost5, heuristic5, 1, [1] ),
        (cost5, heuristic5, 1, [2] ),
        (cost5, heuristic5, 1, [3] ),
        (cost5, heuristic5, 1, [4] ),
        (cost5, heuristic5, 1, [5] ),
        (cost5, heuristic5, 1, [6] ),
        (cost5, heuristic5, 1, [7] ),
        (cost5, heuristic5, 1, [4, 5, 6] ),
        (cost5, heuristic5, 1, [3, 6, 7] ),
        (cost5, heuristic5, 1, [4, 6] ),
        (cost5, heuristic5, 1, [2, 3, 7] ),
        (cost6, heuristic6, 1, [5] )
    ]
    aStarSolns = [
        [1,5,4,7],
        [],
        [],
        [],
        [],
        [start],
        [1, 2],
        
        [1,6,7,9,10],
        [1, 2, 4],
        
        [1, 2, 4, 5],
        [1, 2, 4, 6],
        
        [1],
        [1, 2],
        [1, 2, 3],
        [1, 5, 4],
        [1, 5],
        [1, 2, 6],
        [1, 5, 4, 7],
        [1, 5, 4, 8],
        [1, 5, 9],
        [1, 5, 9, 10],
        [1, 5, 4, 7],
        [1, 2, 3],
        [1, 5],
        [1, 5, 4],
        [1, 2],
        [1, 5, 4, 7],
        [1, 5, 4],
        [1, 5],
        [1],
        [1, 2],
        [1, 7, 3],
        [1, 7, 3, 4],
        [1, 7, 3, 6, 5],
        [1, 7, 3, 6],
        [1, 7],
        [1, 7, 3, 4],
        [1, 7],
        [1, 7, 3, 4],
        [1, 7],
        [1, 2, 3, 5]
        
        ]
    
    dfsTestCases = [
        (cost,start, goals),
        ([],start, goals),
        (cost, 15, goals),
        (cost, start, []),
        (cost2, start, [11]),
        (cost2, start, [6, 7, 10, start]),
        (cost,start, list(range(start + 1, len(cost)))),
        
        (cost1, 1, [10]),
        (cost7, 1, [4]),
        
        (cost3,start, [5]),
        (cost3,start, [6]),   
        
        (cost4, 1, [1]),
        (cost4, 1, [2]),
        (cost4, 1, [3]),
        (cost4, 1, [4]),
        (cost4, 1, [5]),
        (cost4, 1, [6]),
        (cost4, 1, [7]),
        (cost4, 1, [8]),
        (cost4, 1, [9]),
        (cost4, 1, [10]),
        (cost4, 1, [6, 7, 10]),
        (cost4, 1, [3, 4, 7, 10]),
        (cost4, 1, [5, 9, 4]),
        (cost4, 1, [4, 8, 10]),
        (cost4, 1, [2, 8, 5]),
        (cost4, 1, [7, 9, 10]),
        (cost4, 1, [10, 6, 8, 4]),
        (cost4, 1, [9, 7, 5, 10]),
        (cost5, 1, [1]),
        (cost5, 1, [2]),
        (cost5, 1, [3]),
        (cost5, 1, [4]),
        (cost5, 1, [5]),
        (cost5, 1,[6]),
        (cost5, 1,[7]),
        (cost5, 1, [4, 5, 6]),
        (cost5, 1, [3, 6, 7]),
        (cost5, 1, [4, 6]),
        (cost5, 1, [2, 3, 7]),
        (cost6, 1,[5])
        ]
    dfsSolns = [
        [1, 2, 3, 4, 7],
        [],
        [],
        [],
        [],
        [start],
        [1, 2],
        
        [1, 2, 3, 4, 5, 9, 10],
        [1, 2, 4],
        
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5, 6],
        
        [1],
        [1,2],
        [1, 2, 3],
        [1, 2, 3, 4],
        [1, 2, 3, 4, 8, 5],
        [1, 2, 6],
        [1, 2, 3, 4, 7],
        [1, 2, 3, 4, 8],
        [1, 2, 3, 4, 8, 5, 9],
        [1, 2, 3, 4, 8, 5, 9, 10],
        [1, 2, 3, 4, 7],
        [1, 2, 3],
        [1, 2, 3, 4],
        [1, 2, 3, 4],
        [1, 2],
        [1, 2, 3, 4, 7],
        [1, 2, 3, 4],
        [1, 2, 3, 4, 7],
        [1],
        [1, 2],
        [1, 2, 3],
        [1, 2, 3, 4],
        [1, 2, 3, 4, 5],
        [1, 2, 3, 6],
        [1, 7],
        [1, 2, 3, 4],
        [1, 2, 3],
        [1, 2, 3, 4],
        [1, 2],
        [1, 2, 3, 4, 5]
        ]
    
    for i in range(len(aStarTestCases)):
        try:
            if mymodule.A_star_Traversal(*aStarTestCases[i])==aStarSolns[i]:
                print("Test Case", "{0:2d}".format(i), "for  A* Traversal \033[92mPASSED\033[0m")
            else:
                print("Test Case 1 for  A* Traversal \033[91mFAILED\033[0m")
        except Exception as e:
            print("Test Case", "{0:2d}".format(i), "for  A* Traversal \033[91mFAILED\033[0m due to ", e)


        try:
            if mymodule.DFS_Traversal(*dfsTestCases[i])==dfsSolns[i]:
                print("Test Case", "{0:2d}".format(i), "for DFS Traversal \033[92mPASSED\033[0m")
            else:
                print("Test Case", "{0:2d}".format(i), "for DFS Traversal \033[91mFAILED\033[0m")
        except Exception as e:
            print("Test Case", "{0:2d}".format(i), "for DFS Traversal \033[91mFAILED\033[0m due to ", e)

testcase(mymodule)
