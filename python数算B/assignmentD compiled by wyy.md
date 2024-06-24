# Assignment #D: May月考

Updated 1654 GMT+8 May 8, 2024

2024 spring, Complied by ==王诣雅 生命科学学院==





### 编程环境：

操作系统：Windows 11 22H2 22621.3155

Python编程环境：Thonny IDE 4.1.4





## 1. 题目

### 02808: 校门外的树

http://cs101.openjudge.cn/practice/02808/



思路：

按题意写。

代码

```python
L,M=map(int,input().split())
road=[True]*(L+1)
for _ in range(M):
    left,right=map(int,input().split())
    for i in range(left,right+1):
        if road[i]==True:
            road[i]=False
num=0
for j in range(L+1):
    if road[j]==True:
        num+=1
print(num)
```



代码运行截图 ==（至少包含有"Accepted"）==

![image-20240516205555673](C:\Users\30289\AppData\Roaming\Typora\typora-user-images\image-20240516205555673.png)

大概用时：3min



### 20449: 是否被5整除

http://cs101.openjudge.cn/practice/20449/



思路：

按题意写。

代码

```python
thelist=list(map(int,input()))
result=[]
for i in range(len(thelist)):
    newlist=thelist[:(i+1)]
    number=0
    for j in range(len(newlist)):
        number+=(2**j)*newlist[len(newlist)-1-j]
    if number%5==0:
        result.append(1)
    else:
        result.append(0)
result=[str(z) for z in result]
print(''.join(result))

```



代码运行截图 ==（至少包含有"Accepted"）==

![image-20240516213551284](C:\Users\30289\AppData\Roaming\Typora\typora-user-images\image-20240516213551284.png)

大概用时：10min

### 01258: Agri-Net

http://cs101.openjudge.cn/practice/01258/



思路：

Prim算法

代码

```python
from heapq import heappop, heappush


while True:
    try:
        n = int(input())
    except:
        break
    mat= []
    for i in range(n):
        mat.append(list(map(int, input().split())))
    d, v, q, cnt = [100000 for i in range(n)], set(), [], 0
    d[0] = 0
    heappush(q, (d[0], 0))
    while q:
        x, y = heappop(q)
        if y in v:
            continue
        v.add(y)
        cnt += d[y]
        for i in range(n):
            if d[i] > mat[y][i]:
                d[i] = mat[y][i]
                heappush(q, (d[i], i))
    print(cnt) 

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![image-20240517144541445](C:\Users\30289\AppData\Roaming\Typora\typora-user-images\image-20240517144541445.png)

大概用时：40min

### 27635: 判断无向图是否连通有无回路(同23163)

http://cs101.openjudge.cn/practice/27635/



思路：

dfs

代码

```python
n, m = list(map(int, input().split()))
edge = [[]for _ in range(n)]
for _ in range(m):
    a, b = list(map(int, input().split()))
    edge[a].append(b)
    edge[b].append(a)
cnt, flag = set(), False


def dfs(x, y):
    global cnt, flag
    cnt.add(x)
    for i in edge[x]:
        if i not in cnt:
            dfs(i, x)
        elif y != i:
            flag = True


for i in range(n):
    cnt.clear()
    dfs(i, -1)
    if len(cnt) == n:
        break
    if flag:
        break

print("connected:"+("yes" if len(cnt) == n else "no"))
print("loop:"+("yes" if flag else 'no'))

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![image-20240517161254576](C:\Users\30289\AppData\Roaming\Typora\typora-user-images\image-20240517161254576.png)

大概用时：40min



### 27947: 动态中位数

http://cs101.openjudge.cn/practice/27947/



思路：

利用堆。

控制最大堆和最小堆，夹在中间的就是中位数。

代码

```python
import heapq

def dynamic_median(nums):
    min_heap = []  
    max_heap = []  

    median = []
    for i, num in enumerate(nums):
        if not max_heap or num <= -max_heap[0]:
            heapq.heappush(max_heap, -num)
        else:
            heapq.heappush(min_heap, num)

        if len(max_heap) - len(min_heap) > 1:
            heapq.heappush(min_heap, -heapq.heappop(max_heap))
        elif len(min_heap) > len(max_heap):
            heapq.heappush(max_heap, -heapq.heappop(min_heap))

        if i % 2 == 0:
            median.append(-max_heap[0])

    return median

T = int(input())
for _ in range(T):
    nums = list(map(int, input().split()))
    median = dynamic_median(nums)
    print(len(median))
    print(*median)                                             
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![image-20240517195840932](C:\Users\30289\AppData\Roaming\Typora\typora-user-images\image-20240517195840932.png)

大概用时：1h

### 28190: 奶牛排队

http://cs101.openjudge.cn/practice/28190/



思路：

使用单调栈。

输入奶牛的个数以及每只奶牛的身高。遍历每只奶牛，找到每只奶牛的左边界和右边界。找边界是通过单调栈来实现的。最后双层遍历，使最左边的奶牛的右端点值大于最右边的奶牛的位置。更新ans，找到最大，输出。

代码

```python
n=int(input())
height=[int(input()) for _ in range(n)]

left_bound=[-1]*n
right_bound=[n]*n

stack=[]
 
for i in range(n):
    while stack and height[stack[-1]]<height[i]:
        stack.pop()
    if stack:
        left_bound[i]=stack[-1]
    stack.append(i)

stack=[]

for i in range(n-1,-1,-1):
    while stack and height[stack[-1]]>height[i]:
        stack.pop()
    if stack:
        right_bound[i]=stack[-1]
    stack.append(i)
    
ans=0

for i in range(n):
    for j in range(left_bound[i]+1,i):
        if right_bound[j]>i:
            ans=max(ans,i-j+1)
            break
            
print(ans)
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![image-20240517210218454](C:\Users\30289\AppData\Roaming\Typora\typora-user-images\image-20240517210218454.png)

大概用时：1h

## 2. 学习总结和收获

​	前两题签到，中间两题复习了Prim算法和图中的dfs,后两题分别是对最小堆的训练和单调栈的使用。单调栈没练过所以不太做的出来，最小堆时间不够了···中间两题debug用了些时间，还得再熟练啊！模块性强的就是要多复习记忆···为什么会忘呜呜）感觉运气好ac4(如果中间两题一些细节没注意好就完蛋···)继续练习！巩固模板题，加快速度写难题！



