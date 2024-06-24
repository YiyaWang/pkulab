# Assignment #F: All-Killed 满分

Updated 1844 GMT+8 May 20, 2024

2024 spring, Complied by ==王诣雅 生命科学学院==



### 编程环境：

操作系统：Windows 11 22H2 22621.3155

Python编程环境：Thonny IDE 4.1.4



## 1. 题目

### 22485: 升空的焰火，从侧面看

http://cs101.openjudge.cn/practice/22485/



思路：

bfs

对于每一层记录层中节点数，对于当前节点若是每层最后一个，则加入ans。

代码

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



代码运行截图 ==（至少包含有"Accepted"）==

![image-20240526151006134](C:\Users\30289\AppData\Roaming\Typora\typora-user-images\image-20240526151006134.png)

时间：20min



### 28203:【模板】单调栈

http://cs101.openjudge.cn/practice/28203/



思路：

啊···

真心不喜欢单调栈

代码

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



代码运行截图 ==（至少包含有"Accepted"）==

![image-20240526165327367](C:\Users\30289\AppData\Roaming\Typora\typora-user-images\image-20240526165327367.png)

用时：30min



### 09202: 舰队、海域出击！

http://cs101.openjudge.cn/practice/09202/



思路：

拓扑排序。

代码

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



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![image-20240527235654920](C:\Users\30289\AppData\Roaming\Typora\typora-user-images\image-20240527235654920.png)

用时：30min

### 04135: 月度开销

http://cs101.openjudge.cn/practice/04135/



思路：

二分查找（不熟）要加练

代码

```python
def check(x):
    cnt = 0
    cur_sum = 0
    max_sum = 0
    for cost in costs:
        cur_sum += cost
        if cur_sum > x:
            max_sum = max(max_sum, cur_sum - cost)
            cur_sum = cost
            cnt += 1
    max_sum = max(max_sum, cur_sum)
    return cnt < M and max_sum <= x

def find_min_max_expenditure():
    left = max(costs)
    right = sum(costs)
    while left < right:
        mid = (left + right) // 2
        if check(mid):
            right = mid
        else:
            left = mid + 1
    return left

N, M = map(int, input().split())
costs = [int(input()) for _ in range(N)]

result = find_min_max_expenditure()
print(result)
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![image-20240528105520916](C:\Users\30289\AppData\Roaming\Typora\typora-user-images\image-20240528105520916.png)

用时：40min



### 07735: 道路

http://cs101.openjudge.cn/practice/07735/



思路：

经典dijkstra算法。



代码

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



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![image-20240528154630928](C:\Users\30289\AppData\Roaming\Typora\typora-user-images\image-20240528154630928.png)

用时：40min



### 01182: 食物链

http://cs101.openjudge.cn/practice/01182/



思路：

并查集。（啊啊啊啊啊啊啊啊啊啊啊啊能不能考试不考并查集）

代码

```python
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.relation = [0] * n  

    def find(self, x):
        if self.parent[x] != x:
            origin = self.parent[x]
            self.parent[x] = self.find(self.parent[x])
            self.relation[x] = (self.relation[x] + self.relation[origin]) % 3
        return self.parent[x]

    def union(self, x, y, relation):
        rootX = self.find(x)
        rootY = self.find(y)
        if rootX == rootY:
            return (self.relation[y] - self.relation[x] - relation) % 3 == 0
        else:
            self.parent[rootY] = rootX
            self.relation[rootY] = (self.relation[x] + relation - self.relation[y] + 3) % 3
            return True

N, K = map(int, input().split())
uf = UnionFind(N + 1)
false_statements = 0

for _ in range(K):
    D, X, Y = map(int, input().split())
    if X > N or Y > N or (D == 2 and X == Y):
        false_statements += 1
    else:
        if D == 1:
            if not uf.union(X, Y, 0):  
                false_statements += 1
        elif D == 2:
            if not uf.union(X, Y, 1):  
                false_statements += 1

print(false_statements)
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![image-20240528163155871](C:\Users\30289\AppData\Roaming\Typora\typora-user-images\image-20240528163155871.png)

用时：1h+



## 2. 学习总结和收获

尝试手搓食物链失败，看来还是要用模块化的并查集。复习了bfs,dijkstra，拓扑排序，还有二分查找。

并查集和二分查找真的要机考吗···（）

好希望还有作业啊！！！感觉每次写作业都能知道自己还有什么没学的或者不熟的。



