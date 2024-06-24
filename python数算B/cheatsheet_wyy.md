栈

```python
波兰表达式
#样例输入`* + 11.0 12.0 + 24.0 35.0`样例输出`1357.000000`
def calculating(char,x,y):
    if char=='+':
        return (x+y)
    elif char=='-':
        return (x-y)
    elif char=='*':
        return x*y
    elif char=='/' and y!=0:
        return x/y
d=list(input().split())
d=list(reversed(d))#后序表达式不用反过来
nums=[]
signs=[]
for ele in d:
    if ele != '-' and ele !='+' and ele !='*' and ele !='/':
        true_ele=float(ele)
        nums.append(true_ele)
    else:
        one=nums.pop()
        another=nums.pop()
        newnum=calculating(ele,one,another)
        nums.append(newnum)
for result in nums:
    print('%f\n'%result)
#关于栈就是想清楚：遍历什么，遇到什么要入栈，遇到什么不要入栈
```

### 中序表达式转后序表达式

```python
def infix_to_postfix(expression):
    precedence = {'+':1, '-':1, '*':2, '/':2}
    stack = []  
    postfix = [] 
    number = ""  
    
    for char in expression:
        if char.isdigit() or char == '.':  
            number += char
        else:
            if number:  
                postfix.append(number)
                number = ""
            if char in precedence:
                while stack and precedence[char] <= precedence.get(stack[-1], 0):
                    postfix.append(stack.pop())
                stack.append(char)
            elif char == '(':
                stack.append(char)
            elif char == ')':
                while stack and stack[-1] != '(':
                    postfix.append(stack.pop())
                stack.pop()  

    if number:  
        postfix.append(number)

    while stack:
        postfix.append(stack.pop())

    return ' '.join(postfix)

n=int(input())
alll_result=[]
for _ in range(n):
    expression=input()
    alll_result.append(infix_to_postfix(expression))
for en in alll_result:
    print(en)
```

### 合法出栈序列

```python
def is_valid_sequence(x, seq):
    if len(seq) != len(x):
        return "NO"
    stack = []
    pos = 0  
    #以下是精髓
    for char in seq:
        while pos < len(x) and (not stack or stack[-1] != char):
            stack.append(x[pos])
            pos += 1
        if  stack[-1] != char:
            return "NO"
        stack.pop()      
    return "YES"
#以上是精髓
x=input()
result=[]
#第一行是原始字符串x，后面有若干行(不超过50行)，每行一个字符串，所有字符串长度不超过100。用try,except EOFError pass这个结构
try:
    while True:
        expression=input()
        if not expression:
            break
        else:
            result.append(is_valid_sequence(x,expression))
except EOFError:
    pass

for i in result:
    print(i)
```

```python
#卡特兰数：括号匹配：给定 n 对括号，有多少种合法的括号匹配方式。
#二叉树：有 n 个节点的不同二叉树的个数。
#凸多边形：一个凸(n+2)-边形有多少种方式可以划分成 n 个三角形。
#栈操作：给定 n 个元素的入栈顺序，有多少种不同的出栈顺序使得栈在任何时候都不为空。
from math import comb
n=int(input())
print(int(comb(2*n, n)/(n+1)))
```

### 单调栈

```python
#找最大矩形面积 dp+单调栈
m,n=map(int,input().split())
l=[list(map(int,input().split())) for _ in range(m)]
dp=[[0]*(n+1) for _ in range(m)]
for i in range(m):
    for j in range(n):
        if l[i][j]:
            dp[i][j]=0
        else:
            dp[i][j]=dp[i-1][j]+1
ans=0
for i in range(m):
    fl=[[0,0]for _ in range(n+1)]
    stack=[]
    for j in range(n+1):
        ans=max(ans,dp[i][j])
        while stack and dp[i][j]<dp[i][stack[-1]]:
            a=stack.pop()
            fl[a][1]=j
        if stack:
            fl[j][0]=stack[-1]
        else:
            fl[j][0]=-1
        stack.append(j)
    for j in range(n):
        ans=max(ans,(fl[j][1]-fl[j][0]-1)*dp[i][j])
print(ans)  
```

```python
#最长上升子串（大于等于）
def max_increasing_substring(arr):
    n = len(arr)
    if n == 0:
        return []

    max_len = 1
    current_len = 1
    start_index = 0
    best_start = 0

    for i in range(1, n):
        if arr[i] >= arr[i - 1]:#关键就是这里的比较符号
            current_len += 1
        else:
            if current_len > max_len:
                max_len = current_len
                best_start = start_index
            current_len = 1
            start_index = i

    if current_len > max_len:
        max_len = current_len
        best_start = start_index

    return arr[best_start:best_start + max_len]
num = [1,1,2,3,0,0,1,2,5,2,3,9]
print("最大上升子串:", max_increasing_substring(num))
```

## dfs 

```python
def dfs_recursive(graph, start, visited=None):
    if visited is None:
        visited = set()
    visited.add(start)
    print(start)  # 处理节点
    for neighbor in graph[start]:
        if neighbor not in visited:
            dfs_recursive(graph, neighbor, visited)
    return visited

# 示例图（邻接表表示）
graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D', 'E'],
    'C': ['A', 'F'],
    'D': ['B'],
    'E': ['B', 'F'],
    'F': ['C', 'E']
}

# 从节点 'A' 开始 DFS
dfs_recursive(graph, 'A')
```

### 1.棋盘问题

```python
#初始化行数、总棋子数、方案数，棋盘（大一行一列），已访问列表
n,k,ans=0,0,0
chess=[['' for _ in range(10)] for _ in range(10)]
take=[False]*10

#dfs+回溯
def dfs(h,t):
    global ans
    if t==k:
        ans+=1
        return
    if h==n:
        return
    for i in range(h,n):
        for j in range(n):
            if chess[i][j]=='#' and not take[j]:
                take[j]=True
                dfs(i+1,t+1)
                take[j]=False

#多组输入
while True:
    n,k=map(int,input().split())
    if n==-1 and k==-1:
        break
    chess=[list(input()) for _ in range(n)]
    #print(['#.'])print(list('#.'))纯shit
    take=[False]*n
    ans=0
    dfs(0,0)
    print(ans)
```

### 2.最大联通区域面积

```python
dire=[[-1,-1],[-1,0],[-1,1],[1,0],[1,-1],[1,1],[0,1],[0,-1]]

area=0

def dfs(x,y):
    global area
    if matrix[x][y]=='.':
        return
    matrix[x][y]='.'
    area+=1
    for i in range(len(dire)):
        dfs(x+dire[i][0],y+dire[i][1])

T=int(input())
for _ in range(T):
    N,M=map(int,input().split())
    matrix=[['.' for _ in range(M+2)] for _ in range(N+2)]
    for i in range(1,N+1):
        matrix[i][1:-1]=input()
    sur=0
    for i in range(1,N+1):
        for j in range(1,M+1):
            if matrix[i][j]=='W':
                area=0
                dfs(i,j)
                sur=max(sur,area)
    print(sur)
```

### 3.岛屿问题

```python
def dfs_islands(matrix):
    def dfs(x, y):
        if x < 0 or x >= m or y < 0 or y >= n or matrix[x][y] != '1':
            return
        matrix[x][y] = '0'  # 标记为已访问
        for dx, dy in directions:
            dfs(x + dx, y + dy)

    m, n = len(matrix), len(matrix[0])
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    island_count = 0

    for i in range(m):
        for j in range(n):
            if matrix[i][j] == '1':
                island_count += 1
                dfs(i, j)

    return island_count

# 示例矩阵
matrix = [
    ['1', '1', '0', '0'],
    ['1', '0', '0', '1'],
    ['0', '0', '1', '1'],
    ['0', '1', '1', '0']
]

print(dfs_islands(matrix))  # 输出: 2
```

### 4.生成深度优先搜索树

```python
def dfs_generate_tree(graph, root):
    def dfs(node):
        if node in visited:
            return None
        visited.add(node)
        tree_node = TreeNode(node)
        for neighbor in graph[node]:
            child = dfs(neighbor)
            if child:
                tree_node.children.append(child)
        return tree_node

    visited = set()
    return dfs(root)
```

### 5.迷宫问题

```python
def dfs_maze(maze, start, end):
    def dfs(x, y, path):
        if not (0 <= x < m and 0 <= y < n) or maze[x][y] == '#' or (x, y) in visited:
            return False
        path.append((x, y))
        if (x, y) == end:
            return True
        visited.add((x, y))
        for dx, dy in directions:
            if dfs(x + dx, y + dy, path):
                return True
        path.pop()
        return False

    m, n = len(maze), len(maze[0])
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    visited = set()
    path = []

    if dfs(start[0], start[1], path):
        return path
    else:
        return "No path found"

# 示例迷宫
maze = [
    ['.', '.', '.', '#'],
    ['.', '#', '.', '.'],
    ['.', '.', '.', '.'],
    ['#', '.', '#', '.']
]

start = (0, 0)
end = (2, 3)

print(dfs_maze(maze, start, end))  
```

### 6.生成组合和排列

```python
def dfs_combinations(arr, k):
    def dfs(start, path):
        if len(path) == k:
            result.append(path[:])
            return
        for i in range(start, len(arr)):
            path.append(arr[i])
            dfs(i + 1, path)
            path.pop()

    result = []
    dfs(0, [])
    return result

arr = [1, 2, 3, 4]
k = 2
print(dfs_combinations(arr, k))  # 输出: 所有长度为2的组合
```

### 7.算鹰

```python
mat=[list(input()) for _ in range(10)]
dire=[(0,1),(0,-1),(1,0),(-1,0)]
def dfs(x,y):
    if mat[x][y]=='-':
        return
    mat[x][y]='-'
    for dx,dy in dire:
        x2=x+dx
        y2=y+dy
        if 0<=x2<10 and 0<=y2<10 and mat[x2][y2]=='.':
            #注意检查的顺序 先看有没有超出范围再看棋子的状态
            dfs(x2,y2)
result=0
for i in range(10):
    for j in range(10):
        if mat[i][j]=='.':
            result+=1
            dfs(i,j)
print(result)
```

### 8.八皇后

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

### 9.检查图是否有连通和环

```python
from collections import defaultdict
vis=set()
flag=False
#利用父子节点
def dfs(x,y):
    global vis,flag
    vis.add(x)
    for i in edge[x]:
        if i not in vis:
            dfs(i,x)
        elif i!=y:
            flag=True
n,m=map(int,input().split())
edge=defaultdict(list)
for _ in range(m):
    a,b=map(int,input().split())
    edge[a].append(b)
    edge[b].append(a)
dfs(0,-1)
print('connected:yes' if len(vis)==n else 'connected:no')
print('loop:no' if not flag else 'loop:yes')
```

## 二叉树（掌握）

### 1.二叉树的深度

```python
class Treenode:
    def __init__(self,value):
        self.value=value
        self.left=None
        self.right=None

def treedepth(root):
    if root is None:
        return 0
    return 1+max(treedepth(root.left),treedepth(root.right))

n=int(input())
nodes=[Treenode(z+1) for z in range(n)]
for i in range(n):
    node=nodes[i]
    left,right=map(int,input().split())
    if left!=-1 :
        node.left=nodes[left-1]
    if right!=-1:
        node.right=nodes[right-1]

print(treedepth(nodes[0]))
```

### 2.括号嵌套二叉树

```python
class Treenode:
    def __init__(self,value):
        self.value=value
        self.children=[]
        
def parse_tree(s):
    stack=[]
    node=None
    for char in s:
        if char.isalpha():
            node=Treenode(char)
            if stack:
                stack[-1].children.append(node)
        if char=='(':
            if node:
                stack.append(node)
                node=None
        if char==')':
            if stack:
                node=stack.pop()
    return node

def preorder(root):
    if root is None:
        return []
    output=[]
    output.append(root.value)
    for child in root.children:
        output.extend(preorder(child))
    return output
def postorder(root):
    if root is None:
        return []
    output=[]
    for child in root.children:
        output.extend(postorder(child))
    output.append(root.value)
    return output
s=input()
s=''.join(s.split())#注意是字符串格式！
root=parse_tree(s)
print(''.join(preorder(root)))
print(''.join(postorder(root)))
```

### 3.二叉搜索树的层次遍历

![image-20240531231207995](C:\Users\30289\AppData\Roaming\Typora\typora-user-images\image-20240531231207995.png)

```python
class TreeNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

def insert(node, value):
    if node is None:
        return TreeNode(value)
    if value < node.value:
        node.left = insert(node.left, value)
    elif value > node.value:
        node.right = insert(node.right, value)
    return node

def level_order_traversal(root):
    queue = [root]
    traversal = []
    while queue:
        node = queue.pop(0)
        traversal.append(node.value)
        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)
    return traversal#bfs

numbers = list(map(int, input().strip().split()))
numbers = list(dict.fromkeys(numbers))  
root = None
for number in numbers:
    root = insert(root, number)
traversal = level_order_traversal(root)
print(' '.join(map(str, traversal)))
```

### 4.遍历树

请你对输入的树做遍历。遍历的规则是：遍历到每个节点时，按照该节点和所有子节点的值从小到大进行遍历

```python
class Treenode:
    def __init__(self,value):
        self.value=value
        self.children=[]
        
def traversal_print(root,nodes):
    if root.children == []:
        print(root.value)
        return
    pac = {root.value:root}
    for child in root.children:
        pac[child]=nodes[child]
    for value in sorted(pac.keys()):
        if value in root.children:
            traversal_print(pac[value],nodes)
        else:
            print(root.value)#递归

n=int(input())
nodes={}
children_list=[]
for i in range(n):
    info=list(map(int,input().split()))
    nodes[info[0]]=Treenode(info[0])
    for child_value in info[1:]:
        nodes[info[0]].children.append(child_value)
        children_list.append(child_value)
root=nodes[[value for value in nodes.keys() if value not in children_list][0]]
traversal_print(root,nodes)
```

## Dijkstra

### 1.兔子与樱花

```python
import heapq
from heapq import heappop,heappush
import math
P=int(input())
graph={input():[] for _ in range(P)}
Q=int(input())
for _ in range(Q):
    data=list(input().split(' '))
    graph[data[0]].append((data[1],int(data[2])))
    graph[data[1]].append((data[0], int(data[2])))
def Dijkstra(graph,start,end,P):
    if start==end:return []
    dist={i:(math.inf,[]) for i in graph}
    pos=[]
    heappush(pos,(0,start,[]))
    dist[start]=(0,[start])
    while pos:
        dist1,cur,path=heappop(pos)
        for ne,dist2 in graph[cur]:
            if (dist1+dist2)<dist[ne][0]:
                dist[ne]=(dist1+dist2,path+[ne])
                heappush(pos,(dist1+dist2,ne,path+[ne]))
    return dist[end][1]
    
R=int(input())
for _ in range(R):
    start,end=input().split(' ')
    path=Dijkstra(graph,start,end,P)
    s=start
    current=start
    for i in path:
        s+=f'->({[d for (n,d) in graph[current] if n==i][0]})->{i}'
        current=i
    print(s)
```

### 2.道路

变种 除了权重有其他条件限制 重复访问节点不需要vis 也不需要比较

```python
import heapq
K,N,R=int(input()),int(input()),int(input())
graph={i:[] for i in range(1,N+1)}
for _ in range(R):
    S,D,L,T=map(int,input().split())
    graph[S].append((D,L,T))

def dijkstra(graph,n):
    q=[(0,1,0)]
    ans=0
    while q:
        l,cur,cost=heapq.heappop(q)
        if cur==n:
            return l
        for next,nl,nc in graph[cur]:
            if cost+nc<=K:
                heapq.heappush(q,(l+nl,next,nc+cost))
    return -1
print(dijkstra(graph,N))
```

### 3.走山路

bfs+dijkstra

 ```python
 from heapq import heappop,heappush
 import math
 m,n,p=map(int,input().split())
 matrix = []
 for _ in range(m):
     matrix.append(list(input().split()))
 dire=[(0,1),(0,-1),(1,0),(-1,0)]
 
 def bfs(x1,y1):
     q=[]
     v=set()
     heappush(q,(0,x1,y1))
     while q:
         t,x,y=heappop(q)
         v.add((x,y))
         if x==x2 and y==y2:
             return t
         for dx,dy in dire:
             nx=x+dx
             ny=y+dy
             if 0<=nx<m and 0<=ny<n and matrix[nx][ny]!='#' and (nx,ny) not in v:
                 nt=t+abs(int(matrix[x][y])-int(matrix[nx][ny]))
                 heappush(q,(nt,nx,ny))
     return 'NO'
 
 for _ in range(p):
     x1,y1,x2,y2=map(int,input().split())
     if matrix[x1][y1]=='#' or matrix[x2][y2]=='#':
         print('NO')
         continue
     print(bfs(x1,y1))
 ```

## bfs（掌握）

```python
#模板
from collections import deque

def bfs(graph, start):
    queue = deque([start])
    visited = set([start])
    while queue:
        node = queue.popleft()
        print(node)  # 可以在这里处理节点
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

# 示例图（邻接表表示）
graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D', 'E'],
    'C': ['A', 'F'],
    'D': ['B'],
    'E': ['B', 'F'],
    'F': ['C', 'E']
}

# 从节点 'A' 开始 BFS
bfs(graph, 'A')
```

### 1.无权图中最短路径

```python
#bfs：适用于无权图中的最短路径搜索.
#dfs：适用于路径搜索、连通性检测、拓扑排序等。
from collections import deque

def bfs_shortest_path(matrix, start, end):
    m, n = len(matrix), len(matrix[0])
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    queue = deque([(start[0], start[1], 0)])  # (x, y, distance)
    visited = set()
    visited.add((start[0], start[1]))

    while queue:
        x, y, dist = queue.popleft()
        if (x, y) == end:
            return dist

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < m and 0 <= ny < n and matrix[nx][ny] != '#' and (nx, ny) not in visited:
                visited.add((nx, ny))
                queue.append((nx, ny, dist + 1))

    return -1  # 表示无法到达终点

# 示例矩阵
matrix = [
    ['0', '0', '0', '#'],
    ['0', '#', '0', '0'],
    ['0', '0', '0', '0'],
    ['#', '0', '#', '0']
]

start = (0, 0)
end = (2, 3)

print(bfs_shortest_path(matrix, start, end))  # 输出: 5
```

### 2.迷宫问题

```python
def bfs_maze(maze, start, end):
    m, n = len(maze), len(maze[0])
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    queue = deque([(start[0], start[1], 0)])  # (x, y, distance)
    visited = set()
    visited.add((start[0], start[1]))

    while queue:
        x, y, dist = queue.popleft()
        if (x, y) == end:
            return dist

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < m and 0 <= ny < n and maze[nx][ny] != '#' and (nx, ny) not in visited:
                visited.add((nx, ny))
                queue.append((nx, ny, dist + 1))

    return -1  # 表示无法到达终点

# 示例迷宫
maze = [
    ['.', '.', '.', '#'],
    ['.', '#', '.', '.'],
    ['.', '.', '.', '.'],
    ['#', '.', '#', '.']
]

start = (0, 0)
end = (2, 3)

print(bfs_maze(maze, start, end))  # 输出: 5
```

### 3.岛屿问题

```python
def bfs_islands(matrix):
    m, n = len(matrix), len(matrix[0])
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    visited = set()
    islands = 0

    def bfs(x, y):
        queue = deque([(x, y)])
        visited.add((x, y))
        while queue:
            cx, cy = queue.popleft()
            for dx, dy in directions:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < m and 0 <= ny < n and (nx, ny) not in visited and matrix[nx][ny] == '1':
                    visited.add((nx, ny))
                    queue.append((nx, ny))

    for i in range(m):
        for j in range(n):
            if matrix[i][j] == '1' and (i, j) not in visited:
                bfs(i, j)
                islands += 1

    return islands

# 示例矩阵
matrix = [
    ['1', '1', '0', '0'],
    ['1', '0', '0', '1'],
    ['0', '0', '1', '1'],
    ['0', '1', '1', '0']
]

print(bfs_islands(matrix))  # 输出: 3
```

### 4.升空的烟火，从侧面看

```python
class Treenode:
    def __init__(self,value):
        self.left=None
        self.right=None
        self.value=value

n=int(input())
nodes=[Treenode(z+1) for z in range(n)]
for node in nodes:
    left,right=map(int,input().split())
    if left!=-1:
        node.left=nodes[left-1]
    if right!=-1:
        node.right=nodes[right-1]

queue=[]
ans=[]
queue.append(nodes[0])
while queue:
   level_len=(len(queue))
   for i in range(level_len):
       current_node=queue.pop(0)
       if i == level_len-1:
           ans.append(current_node.value)
       if current_node.left is not None:
           queue.append(current_node.left)
       if current_node.right is not None:
           queue.append(current_node.right)
           
ans=[str(z) for z in ans]
print(' '.join(ans))
```

## 最小生成树(prim)（掌握）

### 1.agri_net

```python
import heapq

def prim_heap(n, mat):
    # 初始化最小堆
    min_heap = [(0, 0)]  # (权重, 节点)
    visited = set()
    mst_weight = 0

    while min_heap:
        weight, u = heapq.heappop(min_heap)
        if u in visited:
            continue
        visited.add(u)
        mst_weight += weight

        # 遍历 u 的所有邻居节点
        for v in range(n):
            if v not in visited and mat[u][v] < 100000:  # 避免无限大的权重
                heapq.heappush(min_heap, (mat[u][v], v))

    return mst_weight

while True:
    try:
        n = int(input())
    except:
        break
    mat = []
    for _ in range(n):
        mat.append(list(map(int, input().split())))

    mst_weight = prim_heap(n, mat)
    print(mst_weight)
```

### 2.兔子与星空

```python
import heapq

def prim(graph, start):
    mst = []
    used = set([start])
    edges = [
        (cost, start, to)
        for to, cost in graph[start].items()
    ]
    heapq.heapify(edges)

    while edges:
        cost, frm, to = heapq.heappop(edges)
        if to not in used:
            used.add(to)
            mst.append((frm, to, cost))
            for to_next, cost2 in graph[to].items():
                if to_next not in used:
                    heapq.heappush(edges, (cost2, to, to_next))

    return mst

def solve():
    n = int(input())
    graph = {chr(i+65): {} for i in range(n)}
    for i in range(n-1):
        data = input().split()
        star = data[0]
        m = int(data[1])
        for j in range(m):
            to_star = data[2+j*2]
            cost = int(data[3+j*2])
            graph[star][to_star] = cost
            graph[to_star][star] = cost
    mst = prim(graph, 'A')
    print(sum(x[2] for x in mst))

solve()
```

## 并查集（掌握）

### 1.冰阔落

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

        unique_parents = set(find(x) for x in range(1, n + 1))  # 获取不同集合的根节点
        ans = sorted(unique_parents)  # 输出有冰阔落的杯子编号
        print(len(ans))
        print(*ans)

    except EOFError:
        break
```

## 拓扑排序

```python
from collections import deque, defaultdict

def kahn_topological_sort(graph):
    in_degree = defaultdict(int)
    for u in graph:
        for v in graph[u]:
            in_degree[v] += 1

    queue = deque([u for u in graph if in_degree[u] == 0])
    result = []

    while queue:
        u = queue.popleft()
        result.append(u)
        for v in graph[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)

    if len(result) == len(graph):
        return result
    else:
        return "Graph has a cycle, topological sort not possible"

# 示例图（邻接表表示）
graph = {
    'A': ['C'],
    'B': ['C', 'D'],
    'C': ['E'],
    'D': ['F'],
    'E': ['F'],
    'F': []
}

# 执行拓扑排序
print(kahn_topological_sort(graph))  # 输出: ['A', 'B', 'C', 'D', 'E', 'F'] 或其他合法顺序
```

### 1.舰队、海域出击

```python
from collections import defaultdict

def dfs(p):
    vis[p]=True
    for q in graph[p]:
        in_degree[q]-=1
        if in_degree[q]==0:
            dfs(q)

for _ in range(int(input())):
    n,m=map(int,input().split())
    graph=defaultdict(list)
    vis=[False]*(n+1)
    in_degree=[0]*(n+1)
    for i in range(m):
        x,y=map(int,input().split())
        graph[x].append(y)
        in_degree[y]+=1
    for k in range(1,n+1):
        if in_degree[k]==0 and not vis[k]:
            dfs(k)
    flag=any(not vis[i] for i in range(1,n+1))
    print('Yes' if flag else 'No') 
```

## 【模板】单调栈

```python
n=int(input())
num=list(map(int,input().split()))
ans=[0]*n

stack=[]

for i in range(n):
    while stack and num[i]>num[stack[-1]]:
        ans[stack.pop()]=i+1
    stack.append(i)

while stack:
    ans[stack.pop()]=0

print(*ans)
```

必会题：

```python
#检查嵌套
def check(s):
    stack=[]
    dict={')':'(',']':'[','}':'{'}
    nested=False
    stack=[]
    for char in s:
        if char in '([{':
            stack.append(char)
        if char in ')]}':
            if not stack or stack[-1]!=dict[char]:
                return 'ERROR'
            stack.pop()
            if stack:
                nested=True
    if stack:
        return 'ERROR'
    return 'YES' if nested else 'NO'
s=input()
print(check(s))
```

```python
#pku爱消除
def check(s):
    stack=[]
    for char in s:
        stack.append(char)
        if len(stack)>=3 and stack[-3:]==['P','K','U']:
            stack.pop()
            stack.pop()
            stack.pop()
    return stack
s=input()
print(''.join(check(s)))
```

```python
#感觉好久没做过这么简单的dfs了
class Graphnode:
    def __init__(self,value):
        self.value=value
        self.neighbors=[]

n,m=map(int,input().split())
nodes=[Graphnode(i) for i in range(n)]
for _ in range(m):
    a,b=map(int,input().split())
    nodes[a].neighbors.append(nodes[b])
    nodes[b].neighbors.append(nodes[a])

def dfs(node,result,vis):
    result.append(node.value)
    vis.add(node)
    for nei in node.neighbors:
        if nei not in vis :
           dfs(nei,result,vis)


vis=set()
result=[]
for node in nodes:
    if node not in vis:
        dfs(node,result,vis)
re=[str(z) for z in result]
print(' '.join(re))
```

```python
#拓扑排序
from collections import deque,defaultdict

def min_prizes(n,m,matches):
    graph=defaultdict(list)
    in_degree=[0]*n
    for a,b in matches:
        graph[b].append(a)
        in_degree[a]+=1

    q=deque()
    for i in range(n):
        if in_degree[i]==0:
            q.append(i)
    prizes=[100]*n

    while q:
        current=q.popleft()
        for nei in graph[current]:
            prizes[nei]=max(prizes[nei],prizes[current]+1)
            in_degree[nei]-=1
            if in_degree[nei]==0:
                q.append(nei)
    return sum(prizes)

n,m=map(int,input().split())
matches=[tuple(map(int,input().split())) for _ in range(m)]
print(min_prizes(n,m,matches))
```

```python
#独立完成作业
def find(x):
    if parent[x] != x:
        parent[x] = find(parent[x])
    return parent[x]

def union(x, y):
    root_x = find(x)
    root_y = find(y)
    if root_x != root_y:
        parent[root_y] = root_x

n,m=map(int,input().split())
parent=[i for i in range(n)]
for _ in range(m):
    x,y=map(int,input().split())
    union(x-1,y-1)
good=set(find(i) for i in range(n))
print(len(good))
```

```python
#排队
def find(x):
    if parent[x] != x:
        parent[x] = find(parent[x])
    return parent[x]

def union(x, y):
    root_x = find(x)
    root_y = find(y)
    if root_x != root_y:
        parent[root_y] = root_x
        leader[root_x] = leader[root_y]
T=int(input())
for _ in range(T):
    n,m=map(int,input().split())
    parent=[i for i in range(n+1)]
    leader = [i for i in range(n + 1)]
    for _ in range(m):
        x,y=map(int,input().split())
        union(x,y)
    good=[leader[find(i)] for i in range(1,n+1)]
    print(*good)
```

```python
DFS 适用于无权迷宫路径问题，可以找到一条可行路径，但不保证是最短路径。
Dijkstra 适用于带权图的最短路径问题，可以找到最短路径。
Prim适用于带权图的最小生成树。
```

```python
#牛模板！！！！！！！！！！！杀掉一道prim一道dijkstra
#prim 字典来建图
import heapq

def dijkstra(graph, start):
    pq = [(0, start)]
    dist = {start: 0}
    prev = {start: None}
    while pq:
        current_dist, current_node = heapq.heappop(pq)
        for neighbor, weight in graph[current_node]:
            distance = current_dist + weight
            if neighbor not in dist or distance < dist[neighbor]:
                dist[neighbor] = distance
                prev[neighbor] = current_node
                heapq.heappush(pq, (distance, neighbor))
    return dist, prev

graph = {
    'A': [('B', 1), ('C', 4)],
    'B': [('A', 1), ('C', 2), ('D', 5)],
    'C': [('A', 4), ('B', 2), ('D', 1)],
    'D': [('B', 5), ('C', 1)]
}

start_vertex = 'A'
dist, prev = dijkstra(graph, start_vertex)
print("最短路径距离:", dist)
```

```python
#prim 字典来建图
import heapq

def prim(graph, start):
    mst = []
    visited = set([start])
    edges = [(weight, start, to) for to, weight in graph[start]]
    heapq.heapify(edges)
    
    while edges:
        weight, frm, to = heapq.heappop(edges)
        if to not in visited:
            visited.add(to)
            mst.append((frm, to, weight))
            for next_to, next_weight in graph[to]:
                if next_to not in visited:
                    heapq.heappush(edges, (next_weight, to, next_to))
    
    return mst

graph = {
    'A': [('B', 1), ('C', 4)],
    'B': [('A', 1), ('C', 2), ('D', 5)],
    'C': [('A', 4), ('B', 2), ('D', 1)],
    'D': [('B', 5), ('C', 1)]
}

start_vertex = 'A'
mst = prim(graph, start_vertex)
print("最小生成树的边集合:", mst)
```

```python
#prim 邻接矩阵来建图
import heapq

def prim(adj_matrix):
    n = len(adj_matrix)
    visited = [False] * n
    min_heap = [(0, 0)]  # (cost, node)
    mst_cost = 0
    mst_edges = []

    while min_heap:
        cost, u = heapq.heappop(min_heap)
        if visited[u]:
            continue
        visited[u] = True
        mst_cost += cost
        for v in range(n):
            if not visited[v] and adj_matrix[u][v] != 0:
                heapq.heappush(min_heap, (adj_matrix[u][v], v))
                mst_edges.append((u, v, adj_matrix[u][v]))

    return mst_cost, mst_edges

# 示例邻接矩阵
adj_matrix = [
    [0, 1, 4, 0],
    [1, 0, 2, 5],
    [4, 2, 0, 1],
    [0, 5, 1, 0]
]

mst_cost, mst_edges = prim(adj_matrix)
print("最小生成树的成本:", mst_cost)
print("最小生成树的边:", mst_edges)
```

```python
#dijkstra 邻接矩阵建图
import heapq

def dijkstra(adj_matrix, start):
    n = len(adj_matrix)
    dist = [float('inf')] * n
    dist[start] = 0
    min_heap = [(0, start)]  # (distance, node)
    prev = [-1] * n  # 用于记录路径

    while min_heap:
        current_dist, u = heapq.heappop(min_heap)
        if current_dist > dist[u]:
            continue
        for v in range(n):
            if adj_matrix[u][v] != 0:
                distance = current_dist + adj_matrix[u][v]
                if distance < dist[v]:
                    dist[v] = distance
                    prev[v] = u
                    heapq.heappush(min_heap, (distance, v))

    return dist, prev

# 示例邻接矩阵
adj_matrix = [
    [0, 1, 4, 0],
    [1, 0, 2, 5],
    [4, 2, 0, 1],
    [0, 5, 1, 0]
]

start_vertex = 0
dist, prev = dijkstra(adj_matrix, start_vertex)
print("从起点到各顶点的最短路径距离:", dist)
print("路径的前驱节点:", prev)
```

