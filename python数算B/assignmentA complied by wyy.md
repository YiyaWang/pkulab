# Assignment #A: 图论：算法，树算及栈

Updated 2018 GMT+8 Apr 21, 2024

2024 spring, Complied by ==王诣雅 生命科学学院==



### 编程环境：

操作系统：Windows 11 22H2 22621.3155

Python编程环境：Thonny IDE 4.1.4





## 1. 题目

### 20743: 整人的提词本

http://cs101.openjudge.cn/practice/20743/



思路：

使用栈来处理和反转字符串中括号内的字符。遇到闭括号时，将栈中元素逐个弹出直到开括号，然后将这些元素逆序放回栈中。最后返回栈中所有元素组成的字符串。

代码

```python
def reverse_parentheses(s):
    stack = []
    for char in s:
        if char == ')':
            temp = []
            while stack and stack[-1] != '(':
                temp.append(stack.pop())
            # remove the opening parenthesis
            if stack:
                stack.pop()
            # add the reversed characters back to the stack
            stack.extend(temp)
        else:
            stack.append(char)
    return ''.join(stack)

s = input().strip()
print(reverse_parentheses(s))

```



代码运行截图 ==（至少包含有"Accepted"）==

![image-20240428161336998](C:\Users\30289\AppData\Roaming\Typora\typora-user-images\image-20240428161336998.png)

大概用时：20分钟



### 02255: 重建二叉树

http://cs101.openjudge.cn/practice/02255/



思路：

利用先序和中序遍历结果重建二叉树，并输出后序遍历结果。通过递归将树分为左右子树，然后组合左右子树和根节点的后序遍历结果。

代码

```python
def build_tree(preorder, inorder):
    if not preorder:
        return ''
    
    root = preorder[0]
    root_index = inorder.index(root)
    
    left_preorder = preorder[1:1 + root_index]
    right_preorder = preorder[1 + root_index:]
    
    left_inorder = inorder[:root_index]
    right_inorder = inorder[root_index + 1:]
    
    left_tree = build_tree(left_preorder, left_inorder)
    right_tree = build_tree(right_preorder, right_inorder)
    
    return left_tree + right_tree + root

while True:
    try:
        preorder, inorder = input().split()
        postorder = build_tree(preorder, inorder)
        print(postorder)
    except EOFError:
        break

```



代码运行截图 ==（至少包含有"Accepted"）==

![image-20240428161540681](C:\Users\30289\AppData\Roaming\Typora\typora-user-images\image-20240428161540681.png)

大概用时：30分钟



### 01426: Find The Multiple

http://cs101.openjudge.cn/practice/01426/

要求用bfs实现



思路：

bfs



代码

```python
from collections import deque

def find_multiple(n):
    q = deque()
    q.append((1 % n, "1"))
    visited = set([1 % n])  

    while q:
        mod, num_str = q.popleft()

        if mod == 0:
            return num_str

        for digit in ["0", "1"]:
            new_num_str = num_str + digit
            new_mod = (mod * 10 + int(digit)) % n

            if new_mod not in visited:
                q.append((new_mod, new_num_str))
                visited.add(new_mod)

while True:
    n = int(input())
    if n == 0:
        break
    print(find_multiple(n))
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![image-20240428161940159](C:\Users\30289\AppData\Roaming\Typora\typora-user-images\image-20240428161940159.png)

大概用时：40分钟



### 04115: 鸣人和佐助

bfs, http://cs101.openjudge.cn/practice/04115/



思路：

bfs

用于找到从起点 '@' 到终点 '+' 的最短路径，同时处理特殊障碍 '#' 消耗步数。每次移动更新剩余可用步数和时间。

代码

```python
from collections import deque

M, N, T = map(int, input().split())
graph = [list(input()) for i in range(M)]
direc = [(0,1), (1,0), (-1,0), (0,-1)]
start, end = None, None
for i in range(M):
    for j in range(N):
        if graph[i][j] == '@':
            start = (i, j)
def bfs():
    q = deque([start + (T, 0)])
    visited = [[-1]*N for i in range(M)]
    visited[start[0]][start[1]] = T
    while q:
        x, y, t, time = q.popleft()
        time += 1
        for dx, dy in direc:
            if 0<=x+dx<M and 0<=y+dy<N:
                if (elem := graph[x+dx][y+dy]) == '*' and t > visited[x+dx][y+dy]:
                    visited[x+dx][y+dy] = t
                    q.append((x+dx, y+dy, t, time))
                elif elem == '#' and t > 0 and t-1 > visited[x+dx][y+dy]:
                    visited[x+dx][y+dy] = t-1
                    q.append((x+dx, y+dy, t-1, time))
                elif elem == '+':
                    return time
    return -1
print(bfs())

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![image-20240428162852043](C:\Users\30289\AppData\Roaming\Typora\typora-user-images\image-20240428162852043.png)

大概用时：40分钟



### 20106: 走山路

Dijkstra, http://cs101.openjudge.cn/practice/20106/



思路：

使用优先队列实现带权重的bfs，从给定起点找到终点的最短路径。路径成本是节点间的权重差的绝对值。如果遇到障碍'#'或无法到达则返回'NO'。



代码

```python
from heapq import heappop, heappush

def bfs(x1, y1):
    q = [(0, x1, y1)]
    v = set()
    while q:
        t, x, y = heappop(q)
        v.add((x, y))
        if x == x2 and y == y2:
            return t
        for dx, dy in dir:
            nx, ny = x+dx, y+dy
            if 0 <= nx < m and 0 <= ny < n and ma[nx][ny] != '#' and (nx, ny) not in v:
                nt = t+abs(int(ma[nx][ny])-int(ma[x][y]))
                heappush(q, (nt, nx, ny))
    return 'NO'


m, n, p = map(int, input().split())
ma = [list(input().split()) for _ in range(m)]
dir = [(1, 0), (-1, 0), (0, 1), (0, -1)]
for _ in range(p):
    x1, y1, x2, y2 = map(int, input().split())
    if ma[x1][y1] == '#' or ma[x2][y2] == '#':
        print('NO')
        continue
    print(bfs(x1, y1))

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![image-20240428163107326](C:\Users\30289\AppData\Roaming\Typora\typora-user-images\image-20240428163107326.png)

大概用时：40分钟



### 05442: 兔子与星空

Prim, http://cs101.openjudge.cn/practice/05442/



思路：

构建一个图的最小生成树（MST）。从起点开始，逐步扩展，将最小权重边加入MST，直到覆盖所有顶点。最后计算MST的总权重。

代码

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



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![image-20240428163601164](C:\Users\30289\AppData\Roaming\Typora\typora-user-images\image-20240428163601164.png)

大概用时：40分钟



## 2. 学习总结和收获

复习了栈、bfs等。后面两道题学习了diiskra和prim算法。

通过练习和自学看一些视频会有很多收获！



