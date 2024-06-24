# Assignment #8: 图论：概念、遍历，及 树算

Updated 1919 GMT+8 Apr 8, 2024

2024 spring, Complied by ==王诣雅 生命科学学院==



### 编程环境：

操作系统：Windows 11 22H2 22621.3155

Python编程环境：Thonny IDE 4.1.4





## 1. 题目

### 19943: 图的拉普拉斯矩阵

matrices, http://cs101.openjudge.cn/practice/19943/

请定义Vertex类，Graph类，然后实现



思路：

​	两个类`Vertex`和`Graph`，用于表示和操作无向图。`Vertex`类包含节点的ID和与该节点相连接的其他节点的字典。`Graph`类包含整个图的顶点列表和方法来添加顶点和边。函数`constructLaplacianMatrix`使用输入的节点数和边列表构建图，并生成拉普拉斯矩阵。最后，输入节点和边，打印出生成的拉普拉斯矩阵。



代码

```python
class Vertex:	
    def __init__(self, key):
        self.id = key
        self.connectedTo = {}

    def addNeighbor(self, nbr, weight=0):
        self.connectedTo[nbr] = weight

    def __str__(self):
        return str(self.id) + ' connectedTo: ' + str([x.id for x in self.connectedTo])

    def getConnections(self):
        return self.connectedTo.keys()

    def getId(self):
        return self.id

    def getWeight(self, nbr):
        return self.connectedTo[nbr]

class Graph:
    def __init__(self):
        self.vertList = {}
        self.numVertices = 0

    def addVertex(self, key):
        self.numVertices = self.numVertices + 1
        newVertex = Vertex(key)
        self.vertList[key] = newVertex
        return newVertex

    def getVertex(self, n):
        if n in self.vertList:
            return self.vertList[n]
        else:
            return None

    def __contains__(self, n):
        return n in self.vertList

    def addEdge(self, f, t, weight=0):
        if f not in self.vertList:
            nv = self.addVertex(f)
        if t not in self.vertList:
            nv = self.addVertex(t)
        self.vertList[f].addNeighbor(self.vertList[t], weight)

    def getVertices(self):
        return self.vertList.keys()

    def __iter__(self):
        return iter(self.vertList.values())

def constructLaplacianMatrix(n, edges):
    graph = Graph()
    for i in range(n):	
        graph.addVertex(i)
    
    for edge in edges:	
        a, b = edge
        graph.addEdge(a, b)
        graph.addEdge(b, a)
    
    laplacianMatrix = []	
    for vertex in graph:
        row = [0] * n
        row[vertex.getId()] = len(vertex.getConnections())
        for neighbor in vertex.getConnections():
            row[neighbor.getId()] = -1
        laplacianMatrix.append(row)

    return laplacianMatrix


n, m = map(int, input().split())	
edges = []
for i in range(m):
    a, b = map(int, input().split())
    edges.append((a, b))

laplacianMatrix = constructLaplacianMatrix(n, edges)	

for row in laplacianMatrix:	
    print(' '.join(map(str, row)))

```



代码运行截图 ==（至少包含有"Accepted"）==

![image-20240416141214361](C:\Users\30289\AppData\Roaming\Typora\typora-user-images\image-20240416141214361.png)

大概用时：1hour+



### 18160: 最大连通域面积

matrix/dfs similar, http://cs101.openjudge.cn/practice/18160



思路：

DFS，首先读取矩阵的大小和内容，然后对每个单元格进行检查。如果该单元格包含'W'，启动DFS找到所有相连的'W'单元格，计算连通区域的大小。



代码

```python
dire = [[-1,-1],[-1,0],[-1,1],[0,-1],[0,1],[1,-1],[1,0],[1,1]]

area = 0
def dfs(x,y):
    global area
    if matrix[x][y] == '.':return
    matrix[x][y] = '.'
    area += 1
    for i in range(len(dire)):
        dfs(x+dire[i][0], y+dire[i][1])


for _ in range(int(input())):
    n,m = map(int,input().split())

    matrix = [['.' for _ in range(m+2)] for _ in range(n+2)]
    for i in range(1,n+1):
        matrix[i][1:-1] = input()

    sur = 0
    for i in range(1, n+1):
        for j in range(1, m+1):
            if matrix[i][j] == 'W':
                area = 0 
                dfs(i, j)
                sur = max(sur, area)
    print(sur) 

```



代码运行截图 ==（至少包含有"Accepted"）==

![image-20240416142952281](C:\Users\30289\AppData\Roaming\Typora\typora-user-images\image-20240416142952281.png)

大概用时：1hour+



### sy383: 最大权值连通块

https://sunnywhy.com/sfbj/10/3/383



思路：

依然是DFS。在循环中接收每条边的起点和终点，并将它们添加到`edges`列表中。调用`max_weight`函数并打印结果。



代码

```python
def max_weight(n, m, weights, edges):
    graph = [[] for _ in range(n)]
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)

    visited = [False] * n
    max_weight = 0

    def dfs(node):
        visited[node] = True
        total_weight = weights[node]
        for neighbor in graph[node]:
            if not visited[neighbor]:
                total_weight += dfs(neighbor)
        return total_weight

    for i in range(n):
        if not visited[i]:
            max_weight = max(max_weight, dfs(i))

    return max_weight

n, m = map(int, input().split())
weights = list(map(int, input().split()))
edges = []
for _ in range(m):
    u, v = map(int, input().split())
    edges.append((u, v))

print(max_weight(n, m, weights, edges))

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![image-20240416143502331](C:\Users\30289\AppData\Roaming\Typora\typora-user-images\image-20240416143502331.png)

大概用时：50分钟



### 03441: 4 Values whose Sum is 0

data structure/binary search, http://cs101.openjudge.cn/practice/03441



思路：

使用字典`dict1`记录数组`a`和`b`中所有可能两两之和的频次。随后，程序遍历数组`c`和`d`，检查对于每对元素之和的相反数是否已经在`dict1`中记录过，如果有，则累加这个和在`dict1`中的出现次数到`ans`。

代码

```python
n = int(input())
a = [0]*(n+1)
b = [0]*(n+1)
c = [0]*(n+1)
d = [0]*(n+1)

for i in range(n):
    a[i],b[i],c[i],d[i] = map(int, input().split())

dict1 = {}
for i in range(n):
    for j in range(n):
        if not a[i]+b[j] in dict1:
            dict1[a[i] + b[j]] = 0
        dict1[a[i] + b[j]] += 1

ans = 0
for i in range(n):
    for j in range(n):
        if -(c[i]+d[j]) in dict1:
            ans += dict1[-(c[i]+d[j])]

print(ans)
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![image-20240416144559152](C:\Users\30289\AppData\Roaming\Typora\typora-user-images\image-20240416144559152.png)

大概用时：40分钟



### 04089: 电话号码

trie, http://cs101.openjudge.cn/practice/04089/

Trie 数据结构可能需要自学下。



思路：

字典树。



代码

```python
class TrieNode:
    def __init__(self):
        self.child={}


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, nums):
        curnode = self.root
        for x in nums:
            if x not in curnode.child:
                curnode.child[x] = TrieNode()
            curnode=curnode.child[x]

    def search(self, num):
        curnode = self.root
        for x in num:
            if x not in curnode.child:
                return 0
            curnode = curnode.child[x]
        return 1

t = int(input())
p = []
for _ in range(t):
    n = int(input())
    nums = []
    for _ in range(n):
        nums.append(str(input()))
    nums.sort(reverse=True)
    s = 0
    trie = Trie()
    for num in nums:
        s += trie.search(num)
        trie.insert(num)
    if s > 0:
        print('NO')
    else:
        print('YES')
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![image-20240416145204201](C:\Users\30289\AppData\Roaming\Typora\typora-user-images\image-20240416145204201.png)

大概用时：1hour+



### 04082: 树的镜面映射

http://cs101.openjudge.cn/practice/04082/



思路：

首先构建二叉树，然后执行前序遍历，之后将二叉树转换为n-ary树（要保持结构和父子关系的一致性）。接着，使用BFS遍历n-ary树并输出节点值。

代码

```python
class Noden:
    def __init__(self, value):
        self.child = []
        self.value = value
        self.parent = None

class Node:
    def __init__(self, value):
        self.left = None
        self.right = None
        self.value = value
        self.parent = None

def bfs(noden):
    queue.pop(0)
    out.append(noden.value)
    if noden.child:
        for k in reversed(noden.child):
            queue.append(k)
    if queue:
        bfs(queue[0])
        

def ex(node):
    ans.append(node.value)
    if node.left:
        ex(node.left)
    if node.right:
        ex(node.right)

def reverse(node):
    if node.right == None:
        return node
    else:
        return reverse(node.parent)

def build(s, node, state):
    if not s:
        return
    if state == '0':
        new = Node(s[0][0])
        node.left = new
        new.parent = node
    else:
        pos = reverse(node.parent)
        new = Node(s[0][0])
        pos.right = new
        new.parent = pos
    build(s[1:], new, s[0][1])

def bi_to_n(node):
    if node.left:
        if node.left.value != '$':
            newn = Noden(node.left.value)
            dic[node.left] = newn
            dic[node].child.append(newn)
            newn.parent = dic[node]
            bi_to_n(node.left)
    if node.right:
        if node.right.value != '$':
            newn = Noden(node.right.value)
            dic[node.right] = newn
            dic[node].parent.child.append(newn)
            newn.parent = dic[node].parent
            bi_to_n(node.right)

n = int(input())
k = input().split()
root = Node(k[0][0])
k.pop(0)
if k:
    build(k, root, k[0][1])
ans = []
ex(root)
#print(ans)
dic = {}
dic[root] = Noden(root.value)
bi_to_n(root)
rootn = dic[root]
#print(rootn)
queue = [rootn]
out =[]
bfs(rootn)
print(' '.join(out))

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![image-20240416145539018](C:\Users\30289\AppData\Roaming\Typora\typora-user-images\image-20240416145539018.png)

大概用时：emm several hours···

## 2. 学习总结和收获

感觉对dfs和bfs的认知更深入了。但是vertex、graph、以及新学的Trie需要加强练习。

期中季考试密度太大，练习时间明显减少，参考也增多，等下周考完都补起来！



