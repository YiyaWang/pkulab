# Assignment #7: April 月考

Updated 1557 GMT+8 Apr 3, 2024

2024 spring, Complied by ==王诣雅 生命科学学院==



### 编程环境：

操作系统：Windows 11 22H2 22621.3155

Python编程环境：Thonny IDE 4.1.4





## 1. 题目

### 27706: 逐词倒放

http://cs101.openjudge.cn/practice/27706/



思路：

按照题意写啦~



代码

```python
alist=list(input().split(' '))

x=len(alist)
newlist=[]
for _ in range(x):
    newlist.append(alist.pop())
    
print(' '.join(newlist))
```



代码运行截图 ==（至少包含有"Accepted"）==

![image-20240409133909542](C:\Users\30289\AppData\Roaming\Typora\typora-user-images\image-20240409133909542.png)

大概用时：2分钟



### 27951: 机器翻译

http://cs101.openjudge.cn/practice/27951/



思路：

使用队列，遍历



代码

```python
m,n=map(int,input().split(' '))
wordlist=list(map(int,input().split(' ')))

queue=[]
time=0
for word in wordlist:
    if word not in queue and len(queue)<m:
        queue.append(word)
        time+=1
    elif word not in queue and len(queue)==m:
        queue.pop(0)
        queue.append(word)
        time+=1
    elif word in queue:
        time+=0
        
print(time)
```



代码运行截图 ==（至少包含有"Accepted"）==

![image-20240409135734651](C:\Users\30289\AppData\Roaming\Typora\typora-user-images\image-20240409135734651.png)

大概用时：5分钟



### 27932: Less or Equal

http://cs101.openjudge.cn/practice/27932/



思路：

注意edge cases!!!

代码

```python
n,k=map(int,input().split(' '))
numlist=list(map(int,input().split(' ')))
numlist=sorted(numlist)
if k!=0 and k<n:
    thenum=numlist[k-1]
    nextnum=numlist[k]

    if thenum==nextnum:
        print(-1)
    else:
        print(thenum)
elif k==n:
    print(numlist[k-1])
elif k==0:
    if numlist[0]-1>=1:
        print(numlist[0]-1)
    else:
        print(-1)

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![image-20240409143753886](C:\Users\30289\AppData\Roaming\Typora\typora-user-images\image-20240409143753886.png)

大概用时：15分钟



### 27948: FBI树

http://cs101.openjudge.cn/practice/27948/

思路：

二叉树正常思路。注意递归（防止漏函数名导致反复debug）。

以及要注意建树时初始条件只提供一个节点&后序输出node=None的情况。

总之还是很常规的。

代码

```python
class Node:
    def __init__(self,value):
        self.left=None
        self.right=None
        self.value=value


def treetype(numlist):
    if '0' not in numlist:
        return 'I'
    elif '1' not in numlist:
        return 'B'
    else:
        return 'F'

def buildtree(numlist):
    if len(numlist)==1:
        return Node(treetype(numlist))
    root=Node(treetype(numlist))
    if len(numlist)>1:
        root.left=buildtree(numlist[:(len(numlist)//2)])
        root.right=buildtree(numlist[(len(numlist)//2):])
    return root

def postorder(root):
    if root==None:
        return ''
    output=[]
    output.append(postorder(root.left))
    output.append(postorder(root.right))
    output.append(root.value)
    return ''.join(output)


n=int(input())
alist=list(input())
rootnode=buildtree(alist)
print(postorder(rootnode)) 

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![image-20240409162620500](C:\Users\30289\AppData\Roaming\Typora\typora-user-images\image-20240409162620500.png)

大概用时：20分钟

### 27925: 小组队列

http://cs101.openjudge.cn/practice/27925/



思路：

队列模拟。可以直接调用deque。运用集合。



代码

```python
from collections import deque				

t = int(input())
groups = {}
member_to_group = {}

for _ in range(t):
    members = list(map(int, input().split()))
    group_id = members[0]  
    groups[group_id] = deque()
    for member in members:
        member_to_group[member] = group_id

queue = deque()
queue_set = set()


while True:
    command = input().split()
    if command[0] == 'STOP':
        break
    elif command[0] == 'ENQUEUE':
        x = int(command[1])
        group = member_to_group.get(x, None)
        if group is None:
            group = x
            groups[group] = deque([x])
            member_to_group[x] = group
        else:
            groups[group].append(x)
        if group not in queue_set:
            queue.append(group)
            queue_set.add(group)
    elif command[0] == 'DEQUEUE':
        if queue:
            group = queue[0]
            x = groups[group].popleft()
            print(x)
            if not groups[group]:  
                queue.popleft()
                queue_set.remove(group)
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![image-20240409173534907](C:\Users\30289\AppData\Roaming\Typora\typora-user-images\image-20240409173534907.png)

大概用时：最后做的，已经超时了···



### 27928: 遍历树

http://cs101.openjudge.cn/practice/27928/



思路：

精髓是找爸爸找儿子。。。建立映射关系。对父节点和子节点排序遍历输出。



代码

```python
class TreeNode:
    def __init__(self, value):
        self.value = value
        self.children = []


def traverse_print(root, nodes):
    if root.children == []:
        print(root.value)
        return
    pac = {root.value: root}
    for child in root.children:
        pac[child] = nodes[child]
    for value in sorted(pac.keys()):
        if value in root.children:
            traverse_print(pac[value], nodes)
        else:
            print(root.value)


n = int(input())
nodes = {}
children_list = []
for i in range(n):
    info = list(map(int, input().split()))
    nodes[info[0]] = TreeNode(info[0])
    for child_value in info[1:]:
        nodes[info[0]].children.append(child_value)
        children_list.append(child_value)
root = nodes[[value for value in nodes.keys() if value not in children_list][0]]
traverse_print(root, nodes)
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![image-20240409172648880](C:\Users\30289\AppData\Roaming\Typora\typora-user-images\image-20240409172648880.png)

大概用时：40分钟

## 2. 学习总结和收获

前三题签到题。第四题是很常规的二叉树，注意一些细节，完善了一会也ac了。然后时间有点不太够。。。自己计时考的，感觉只能ac四题，sad。期中季太忙了···等考完期中一定给数算多花时间ww！（感觉相比二叉树，多子树掌握得一般般···得多多针对练习。）小组队列有所参考，希望下次能有时间。冲冲冲希望能ac更多。

感觉每次写作业or月考都能学到好多&复习完善好多！





