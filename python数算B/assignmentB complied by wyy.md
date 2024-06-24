# Assignment #B: 图论和树算

Updated 1709 GMT+8 Apr 28, 2024

2024 spring, Complied by ==同学的姓名、院系==



### 编程环境：

操作系统：Windows 11 22H2 22621.3155

Python编程环境：Thonny IDE 4.1.4





## 1. 题目

### 28170: 算鹰

dfs, http://cs101.openjudge.cn/practice/28170/



思路：

dfs计算连通的点的个数。

代码

```python
def dfs(x,y):
    graph[x][y] = "-"
    for dx,dy in [(1,0),(-1,0),(0,1),(0,-1)]:
        if 0<=x+dx<10 and 0<=y+dy<10 and graph[x+dx][y+dy] == ".":
            dfs(x+dx,y+dy)
            
graph = []
result = 0
for i in range(10):
    graph.append(list(input()))
for i in range(10):
    for j in range(10):
        if graph[i][j] == ".":
            result += 1
            dfs(i,j)
            
print(result)
```



代码运行截图 ==（至少包含有"Accepted"）==

![image-20240506155114199](C:\Users\30289\AppData\Roaming\Typora\typora-user-images\image-20240506155114199.png)

大概用时：30min

### 02754: 八皇后

dfs, http://cs101.openjudge.cn/practice/02754/



思路：

dfs+回溯

代码

```python
def is_safe(board, row, col):
    # 检查当前位置是否安全
    # 检查同一列是否有皇后
    for i in range(row):
        if board[i][col] == 1:
            return False
    # 检查左上方是否有皇后
    i = row - 1
    j = col - 1
    while i >= 0 and j >= 0:
        if board[i][j] == 1:
            return False
        i -= 1
        j -= 1
    # 检查右上方是否有皇后
    i = row - 1
    j = col + 1
    while i >= 0 and j < 8:
        if board[i][j] == 1:
            return False
        i -= 1
        j += 1
    return True

def solve_n_queens(board, row, b, count):
    # 递归回溯求解八皇后问题
    if row == 8:
        # 找到一个解，将解添加到结果列表
        count[0] += 1
        if count[0] == b:
            return int(''.join(str(board[i].index(1) + 1) for i in range(8)))
        return None
    for col in range(8):
        if is_safe(board, row, col):
            # 当前位置安全，放置皇后
            board[row][col] = 1
            # 继续递归放置下一行的皇后
            result = solve_n_queens(board, row + 1, b, count)
            if result is not None:
                return result
            # 回溯，撤销当前位置的皇后
            board[row][col] = 0
    return None

n = int(input())
for _ in range(n):
    b = int(input())
    board = [[0] * 8 for _ in range(8)]
    count = [0]
    solution = solve_n_queens(board, 0, b, count)
    print(solution)

```



代码运行截图 ==（至少包含有"Accepted"）==

![image-20240506164407066](C:\Users\30289\AppData\Roaming\Typora\typora-user-images\image-20240506164407066.png)

大概用时：40min



### 03151: Pots

bfs, http://cs101.openjudge.cn/practice/03151/



思路：

bfs队列保存待搜索的状态，逐层扩展，直到找到目标状态或者队列为空。

代码

```python
def bfs(A, B, C):
    start = (0, 0)
    visited = set()
    visited.add(start)
    queue = [(start, [])]

    while queue:
        (a, b), actions = queue.pop(0)

        if a == C or b == C:
            return actions

        next_states = [(A, b), (a, B), (0, b), (a, 0), (min(a + b, A),\
                max(0, a + b - A)), (max(0, a + b - B), min(a + b, B))]

        for i in next_states:
            if i not in visited:
                visited.add(i)
                new_actions = actions + [get_action(a, b, i)]
                queue.append((i, new_actions))

    return ["impossible"]


def get_action(a, b, next_state):
    if next_state == (A, b):
        return "FILL(1)"
    elif next_state == (a, B):
        return "FILL(2)"
    elif next_state == (0, b):
        return "DROP(1)"
    elif next_state == (a, 0):
        return "DROP(2)"
    elif next_state == (min(a + b, A), max(0, a + b - A)):
        return "POUR(2,1)"
    else:
        return "POUR(1,2)"


A, B, C = map(int, input().split())
solution = bfs(A, B, C)

if solution == ["impossible"]:
    print(solution[0])
else:
    print(len(solution))
    for i in solution:
        print(i)
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![image-20240506164748648](C:\Users\30289\AppData\Roaming\Typora\typora-user-images\image-20240506164748648.png)

大概用时：40min



### 05907: 二叉树的操作

http://cs101.openjudge.cn/practice/05907/



思路：

建树，根据给定的操作进行交换或查找树的根节点。

代码

```python
def swap(x, y):
    tree[loc[x][0]][loc[x][1]] = y
    tree[loc[y][0]][loc[y][1]] = x
    loc[x], loc[y] = loc[y], loc[x]


for _ in range(int(input())):
    n, m = map(int, input().split())
    tree = {}
    loc = [[] for _ in range(n)]
    for _ in range(n):
        a, b, c = map(int, input().split())
        tree[a] = [b, c]
        loc[b], loc[c] = [a, 0], [a, 1]
    for _ in range(m):
        op = list(map(int, input().split()))
        if op[0] == 1:
            swap(op[1], op[2])
        else:
            cur = op[1]
            while tree[cur][0] != -1:
                cur = tree[cur][0]
            print(cur)

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![image-20240506165204919](C:\Users\30289\AppData\Roaming\Typora\typora-user-images\image-20240506165204919.png)

大概用时：30min



### 18250: 冰阔落 I

Disjoint set, http://cs101.openjudge.cn/practice/18250/



思路：

并查集，通过 `find` 函数寻找元素的根节点，并使用路径压缩进行优化，`union` 函数将两个集合合并。然后根据输入的边的信息，判断是否构成环，最后输出不相交集合的个数和各个集合的元素。

代码

```python
def find(x):
    if parent[x] != x:
        parent[x] = find(parent[x])
    return parent[x]

def union(x, y):
    root_x = find(x)
    root_y = find(y)
    if root_x != root_y:
        parent[root_y] = root_x

while True:
    try:
        n, m = map(int, input().split())
        parent = list(range(n + 1))

        for _ in range(m):
            a, b = map(int, input().split())
            if find(a) == find(b):
                print('Yes')
            else:
                print('No')
                union(a, b)

        unique_parents = set(find(x) for x in range(1, n + 1))  
        ans = sorted(unique_parents)  
        print(len(ans))
        print(*ans)

    except EOFError:
        break
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![image-20240506165529739](C:\Users\30289\AppData\Roaming\Typora\typora-user-images\image-20240506165529739.png)

大概用时：40min



### 05443: 兔子与樱花

http://cs101.openjudge.cn/practice/05443/



思路：

Dijkstra 算法求解最短路径。首先构建了一个图，然后根据输入的边的信息，计算出从起点到终点的最短路径，并输出路径和路径上的边的权重。

代码

```python
import heapq
import math
def dijkstra(graph,start,end,P):
    if start == end: return []
    dist = {i:(math.inf,[]) for i in graph}
    dist[start] = (0,[start])
    pos = []
    heapq.heappush(pos,(0,start,[]))
    while pos:
        dist1,current,path = heapq.heappop(pos)
        for (next,dist2) in graph[current].items():
            if dist2+dist1 < dist[next][0]:
                dist[next] = (dist2+dist1,path+[next])
                heapq.heappush(pos,(dist1+dist2,next,path+[next]))
    return dist[end][1]

P = int(input())
graph = {input():{} for _ in range(P)}
for _ in range(int(input())):
    place1,place2,dist = input().split()
    graph[place1][place2] = graph[place2][place1] = int(dist)

for _ in range(int(input())):
    start,end = input().split()
    path = dijkstra(graph,start,end,P)
    s = start
    current = start
    for i in path:
        s += f'->({graph[current][i]})->{i}'
        current = i
    print(s)
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![image-20240506165833575](C:\Users\30289\AppData\Roaming\Typora\typora-user-images\image-20240506165833575.png)

大概用时：40min



## 2. 学习总结和收获

感觉要机考了，得找时间开始刷题···





