# Assignment #5: "树"算：概念、表示、解析、遍历

Updated 2124 GMT+8 March 17, 2024

2024 spring, Complied by ==王诣雅 生命科学学院==



### 编程环境：

操作系统：Windows 11 22H2 22621.3155

Python编程环境：Thonny IDE 4.1.4



## 1. 题目

### 27638: 求二叉树的高度和叶子数目

http://cs101.openjudge.cn/practice/27638/



思路：

先初始化一个数/节点。然后用递归的思想定义求树高和叶数的函数（以node为变量）。创建n个节点并标记，根据输入的index连接节点建树。利用index找到根节点，运用函数求出树高度和叶数。



代码

```python
class Treenode:
    def __init__(self):
        self.left=None
        self.right=None

def treeheight(node):
    if node is None:
        return -1
    return max(treeheight(node.left),treeheight(node.right))+1
    
def leafcount(node):
    if node is None:
        return 0
    elif node.left is None and node.right is None:
        return 1
    else:
        return leafcount(node.left)+leafcount(node.right)

n=int(input())
nodes=[Treenode()for _ in range(n)]
has_parent=[False]*n

for i in range(n):
    left_index,right_index=map(int,input().split())
    if left_index!=-1:
        nodes[i].left=nodes[left_index]
        has_parent[left_index]=True
    if right_index!=-1:
        nodes[i].right=nodes[right_index]
        has_parent[right_index]=True
    
root_node=nodes[has_parent.index(False)]
h=treeheight(root_node)
num=leafcount(root_node)

print(f'{h} {num}') 

```



代码运行截图 ==（至少包含有"Accepted"）==

![image-20240325223715726](C:\Users\30289\AppData\Roaming\Typora\typora-user-images\image-20240325223715726.png)

用时：20分钟



### 24729: 括号嵌套树

http://cs101.openjudge.cn/practice/24729/



思路：

注意到不是二叉树。所以建树时要注意。然后定义三个函数。第一个函数是解析树找到根节点。方法与之前栈的练习一样。第二个函数是前序输出树，第三个函数是后序输出树，都运用递归的思想。最后加上主函数。



代码

```python
class Treenode():
    def __init__(self,value):
        self.value=value
        self.children=[]

def root(s):
    stack=[]
    node=None
    for char in s:
        if char.isalpha():
            node=Treenode(char)
            if stack:
                stack[-1].children.append(node)
            else:
                stack.append(node)
        elif char=='(':
            if node:
                stack.append(node)
                node=None
        elif char==')':
            node=stack.pop()
        
    return node

def preorder(node):
    result=[node.value]
    for child in node.children:
        result.extend(preorder(child))
    return ''.join(result)

def postorder(node):
    result=[]
    for child in node.children:
        result.extend(postorder(child))
    result.append(node.value)
    return ''.join(result)

s=input().strip()
s=''.join(s.split())
node=root(s)

print(preorder(node))
print(postorder(node))

```



代码运行截图 ==（至少包含有"Accepted"）==

![image-20240326143319271](C:\Users\30289\AppData\Roaming\Typora\typora-user-images\image-20240326143319271.png)

大概用时：20分钟

### 02775: 文件结构“图”

http://cs101.openjudge.cn/practice/02775/



思路：

先建立节点模拟目录，然后根据题目逻辑连接子节点即子目录和子文件，最后遍历输出目录、子目录和子文件。

代码

```python
class Node:
    def __init__(self,name):
        self.name=name
        self.dirs=[]
        self.files=[]

def print_(root,m):
    pre='|     '*m
    print(pre+root.name)
    for Dir in root.dirs:
        print_(Dir,m+1)
    for file in sorted(root.files):
        print(pre+file)
        
tests,test=[],[]
while True:
    s=input()
    if s=='#':
        break
    elif s=='*':
        tests.append(test)
        test=[]
    else:
        test.append(s)
for n,test in enumerate(tests,1):
    root=Node('ROOT')
    stack=[root]
    print(f'DATA SET {n}:')
    for i in test:
        if i[0]=='d':
            Dir=Node(i)
            stack[-1].dirs.append(Dir)
            stack.append(Dir)
        elif i[0]=='f':
            stack[-1].files.append(i)
        else:
            stack.pop()
    print_(root,0)
    print()

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![image-20240326211719650](C:\Users\30289\AppData\Roaming\Typora\typora-user-images\image-20240326211719650.png)

大概用时：···遇到长得丑的题目就不太行了，于是参考了一把qwq，时间为薛定谔用时（x min)

### 25140: 根据后序表达式建立队列表达式

http://cs101.openjudge.cn/practice/25140/



思路：

先建立树节点。然后第一个函数根据后序表达式找到根节点，第二个函数根据根节点遍历输出树，即队列表达式。

代码

```python
class Treenode:
    def __init__(self,value):
        self.value=value
        self.left=None
        self.right=None

def buildtree(post):
    stack=[]
    for char in post:
        node=Treenode(char)
        if char.isupper():
            node.right=stack.pop()
            node.left=stack.pop()
        stack.append(node)
    return stack[0]

def row(root):
    queue=[root]
    trans=[]
    while queue:
        node=queue.pop(0)
        trans.append(node.value)
        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)
    return ''.join(trans)

n=int(input().strip())
for _ in range(n):
    post=input().strip()
    root=buildtree(post)
    print(row(root)[::-1])
        
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![image-20240326211216042](C:\Users\30289\AppData\Roaming\Typora\typora-user-images\image-20240326211216042.png)

大概用时：20分钟

### 24750: 根据二叉树中后序序列建树

http://cs101.openjudge.cn/practice/24750/



思路：

先建立二叉树节点。定义两个函数。第一个是以中序遍历序列和后序遍历序列为变量来在递归中建立树最后返回根节点。第二个是前序遍历输出树的函数。最后加上主函数。要注意递归中建立左子树和右子树时对原来列表的分割要仔细。

代码

```python
class Treenode():
    def __init__(self,value):
        self.left=None
        self.right=None
        self.value=value
        

def buildtree(inorder,postorder):
    if not inorder or not postorder:
        return None
    root=postorder.pop()
    rootnode=Treenode(root)
    
    root_index=inorder.index(root)
    rootnode.left=buildtree(inorder[:root_index],postorder[:root_index])
    rootnode.right=buildtree(inorder[root_index+1:],postorder[root_index:])
    
    return rootnode
        
def preorder(node):
    if node is None:
        return []
    result=[node.value]
    result.extend(preorder(node.left))
    result.extend(preorder(node.right))
    return ''.join(result)


inorder=list(input().strip())
postorder=list(input().strip())

thenode=buildtree(inorder,postorder)
print(preorder(thenode))

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![image-20240326164918160](C:\Users\30289\AppData\Roaming\Typora\typora-user-images\image-20240326164918160.png)

大概用时：30分钟



### 22158: 根据二叉树前中序序列建树

http://cs101.openjudge.cn/practice/22158/



思路：

跟上面的题目类似。要注意的是：1.pop（0）会改变列表。2.注意循环输出。



代码

```python
class Treenode():
    def __init__(self,value):
        self.left=None
        self.right=None
        self.value=value
        

def buildtree(preorder,inorder):
    if not preorder or not inorder:
        return None
    root=preorder[0]
    rootnode=Treenode(root)
    
    root_index=inorder.index(root)
    rootnode.left=buildtree(preorder[1:root_index+1],inorder[:root_index])
    rootnode.right=buildtree(preorder[root_index+1:],inorder[root_index+1:])
    
    return rootnode
        
def postorder(node):
    if node is None:
        return []
    result=[]
    result.extend(postorder(node.left))
    result.extend(postorder(node.right))
    result.append(node.value)
    return ''.join(result)

while True:
    try:
        preorder=input().strip()
        inorder=input().strip()
        preorder=list(preorder)
        inorder=list(inorder)
        print(postorder(buildtree(preorder,inorder)))
    except EOFError:
        break

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![image-20240326182333451](C:\Users\30289\AppData\Roaming\Typora\typora-user-images\image-20240326182333451.png)

大概用时：30分钟



## 2. 学习总结和收获

1.对树祛魅，其实没有想象得那么难，挺有意思的，格式化很强，递归的思想很多。之后要多做点递归的题。

2.对于长得丑的题目还是会发怵···有点看不下去。准备通过多做此类题来改善。





