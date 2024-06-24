# Assignment #6: "树"算：Huffman,BinHeap,BST,AVL,DisjointSet

Updated 2214 GMT+8 March 24, 2024

2024 spring, Complied by ==王诣雅 生命科学学院==



### 编程环境：

操作系统：Windows 11 22H2 22621.3155

Python编程环境：Thonny IDE 4.1.4



## 1. 题目

### 22275: 二叉搜索树的遍历

http://cs101.openjudge.cn/practice/22275/



思路：

先建二叉树的类。根据前序表达式递归建树(观察考虑到列表第一个数是根节点，之后比根节点小的数都是其左子树，比根节点大的数都是其右子树），定义后续表达式递归输出树的函数。



代码

```python
class BinaryTree():
    def __init__(self, key):
        self.key = key
        self.leftchild = None
        self.rightchild = None

def buildtree(s):
    if len(s) == 0:
        return None
    node = BinaryTree(s[0])
    idx = len(s)
    for i in range(1, len(s)):
        if s[i] > s[0]:
            idx = i
            break
    node.leftchild = buildtree(s[1:idx])
    node.rightchild = buildtree(s[idx:])
    return node

def postorder(tree):
    if tree is None:
        return []
    result = []
    result.extend(postorder(tree.leftchild))
    result.extend(postorder(tree.rightchild))
    result.append(tree.key) 
    return result

n = int(input())
prelist = list(map(int, input().split()))
node = buildtree(prelist)
re = postorder(node)
print(' '.join(map(str, re)))  
```



代码运行截图 ==（至少包含有"Accepted"）==

![image-20240402151805958](C:\Users\30289\AppData\Roaming\Typora\typora-user-images\image-20240402151805958.png)

大概用时：30分钟



### 05455: 二叉搜索树的层次遍历

http://cs101.openjudge.cn/practice/05455/



思路：

创建二叉树类。先根据输入的无序表建BST（定义insert递归函数），然后再定义一个函数层次遍历输出BFS该BST（利用队列）。



代码

```python
class Treenode:
    def __init__(self,value):
        self.value=value
        self.left=None
        self.right=None

def insert(node,value):
    if node is None:
        return Treenode(value)
    if value<node.value:
        node.left=insert(node.left,value)
    elif value>node.value:
        node.right=insert(node.right,value)
    return node

def level_order_traversal(root):
    queue=[root]
    traversal=[]
    while queue:
        node=queue.pop(0)
        traversal.append(node.value)
        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)
    return traversal

numbers=list(map(int,input().strip().split()))
numbers=list(dict.fromkeys(numbers))
root=None
for number in numbers:
    root=insert(root,number)
traversal=level_order_traversal(root)
print(' '.join(map(str,traversal)))

```



代码运行截图 ==（至少包含有"Accepted"）==

![image-20240402163619718](C:\Users\30289\AppData\Roaming\Typora\typora-user-images\image-20240402163619718.png)

大概用时：40分钟



### 04078: 实现堆结构

http://cs101.openjudge.cn/practice/04078/

练习自己写个BinHeap。当然机考时候，如果遇到这样题目，直接import heapq。手搓栈、队列、堆、AVL等，考试前需要搓个遍。



思路：

手搓了一遍bh。。。

输出时类型转换要注意。



代码

```python
class Binheap:
    def __init__(self):
        self.heaplist=[0]
        self.currentsize=0
    def percup(self,i):
        while i // 2 > 0:
            if self.heaplist[i]<self.heaplist[i//2]:
                tmp=self.heaplist[i//2]
                self.heaplist[i//2]=self.heaplist[i]
                self.heaplist[i]=tmp
            i=i//2
    def insert(self,k):
        self.heaplist.append(k)
        self.currentsize=self.currentsize+1
        self.percup(self.currentsize)
    def percdown(self,i):
        while(i*2)<=self.currentsize:
            mc=self.minchild(i)
            if self.heaplist[i]>self.heaplist[mc]:
                tmp=self.heaplist[i]
                self.heaplist[i]=self.heaplist[mc]
                self.heaplist[mc]=tmp
            i=mc
    def minchild(self,i):
        if i*2+1>self.currentsize:
            return i*2
        else:
            if self.heaplist[i*2]<self.heaplist[i*2+1]:
                return i*2
            else:
                return i*2+1
    def delmin(self):
        retval=self.heaplist[1]
        self.heaplist[1]=self.heaplist[self.currentsize]
        self.currentsize=self.currentsize-1
        self.heaplist.pop()
        self.percdown(1)
        return retval
    def buildheap(self,alist):
        i=len(alist)//2
        self.currentsize=len(alist)
        self.heaplist=[0]+alist[:]
        while(i>0):
            self.percdown(i)
            i=i-1
n=int(input())
bh=Binheap()
for _ in range(n):
    inp=input().strip()
    if inp[0]=='1':
        bh.insert(int(inp.split()[1]))
    else:
        print(bh.delmin())

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![image-20240402181125455](C:\Users\30289\AppData\Roaming\Typora\typora-user-images\image-20240402181125455.png)

大概用时：20分钟



### 22161: 哈夫曼编码树

http://cs101.openjudge.cn/practice/22161/



思路：

建树：主要利用最小堆，每次取出weight最小的两个节点，weight相加后创建节点，连接左右孩子，再入堆，直至堆中只剩一个节点.

编码：跟踪每一步走的是左还是右，用0和1表示，直至遇到有char值的节点，说明到了叶子节点，将01字串添加进字典.

解码：根据01字串决定走左还是右，直至遇到有char值的节点，将char值取出.



代码

```python
import heapq

class Node:
    def __init__(self,weight,char=None):
        self.weight=weight
        self.char=char
        self.left=None
        self.right=None
    
    def __lt__(self,other):
        if self.weight==other.weight:
            return self.char<other.char
        return self.weight<other.weight
    
def build_huffman_tree(characters):
    heap=[]
    for char,weight in characters.items():
        heapq.heappush(heap,Node(weight,char))
    while len(heap)>1:
        left=heapq.heappop(heap)
        right=heapq.heappop(heap)
        merged=Node(left.weight+right.weight,min(left.char,right.char))
        merged.left=left
        merged.right=right
        heapq.heappush(heap,merged)
    return heap[0]
def encode_huffman_tree(root):
    codes={}
    def traverse(node,code):
        if node.left is None and node.right is None:
            codes[node.char]=code
        else:
            traverse(node.left,code+'0')
            traverse(node.right,code+'1')
    traverse(root,'')
    return codes
def huffman_encoding(codes,string):
    encoded=''
    for char in string:
        encoded+=codes[char]
    return encoded
def huffman_decoding(root,encoded_string):
    decoded=''
    node=root
    for bit in encoded_string:
        if bit=='0':
            node=node.left
        else:
            node=node.right
        if node.left is None and node.right is None:
            decoded+=node.char
            node=root
    return decoded

n=int(input())
characters={}
for _ in range(n):
    char,weight=input().split()
    characters[char]=int(weight)
    
#build huffman tree
huffman_tree=build_huffman_tree(characters)
#encode and decode
codes=encode_huffman_tree(huffman_tree)

strings=[]
while True:
    try:
        line=input()
        strings.append(line)
    except EOFError:
        break
results=[]
for string in strings:
    if string[0] in ('0','1'):
        results.append(huffman_decoding(huffman_tree,string))
    else:
        results.append(huffman_encoding(codes,string))
for result in results:
    print(result) 

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![image-20240402212950077](C:\Users\30289\AppData\Roaming\Typora\typora-user-images\image-20240402212950077.png)

大概用时：1h+



### 晴问9.5: 平衡二叉树的建立

https://sunnywhy.com/sfbj/9/5/359



思路：

1. 读取输入序列。
2. 将值插入到一个AVL树中。AVL树是一个自平衡的二叉搜索树，任何节点的两个子树的高度最多相差一。
3. 执行AVL树的先序遍历并打印结果。

代码：

```python
class Node:
    def __init__(self,value):
        self.value=value
        self.left=None
        self.right=None
        self.height=1

class AVL:
    def __init__(self):
        self.root=None
        
    def insert(self,value):
        if not self.root:
            self.root=Node(value)
        else:
            self.root=self._insert(value,self.root)
    def _insert(self,value,node):
        if not node:
            return Node(value)
        elif value<node.value:
            node.left=self._insert(value,node.left)
        else:
            node.right=self._insert(value,node.right)
        node.height=1+max(self._get_height(node.left),self._get_height(node.right))
        balance=self._get_balance(node)
        if balance>1:
            if value<node.left.value:
                return self._rotate_right(node)
            else:
                node.left=self._rotate_left(node.left)
                return self._rotate_right(node)
        if balance<-1:
            if value>node.right.value:
                return self._rotate_left(node)
            else:
                node.right=self._rotate_right(node.right)
                return self._rotate_left(node)
        return node
    def _get_height(self,node):
        if not node:
            return 0
        return node.height
    def _get_balance(self,node):
        if not node:
            return 0
        return self._get_height(node.left)-self._get_height(node.right)
    def _rotate_left(self,z):
        y=z.right
        T2=y.left
        y.left=z
        z.right=T2
        z.height=1+max(self._get_height(z.left),self._get_height(z.right))
        y.height=1+max(self._get_height(y.left),self._get_height(y.right))
        return y
    def _rotate_right(self,y):
        x=y.left
        T2=x.right
        x.right=y
        y.left=T2
        y.height=1+max(self._get_height(y.left),self._get_height(y.right))
        x.height=1+max(self._get_height(x.left),self._get_height(x.right))
        return x
    def preorder(self):
        return self._preorder(self.root)
    def _preorder(self,node):
        if not node:
            return []
        return [node.value]+self._preorder(node.left)+self._preorder(node.right)
n=int(input().strip())
sequence=list(map(int,input().strip().split()))
avl=AVL()
for value in sequence:
    avl.insert(value)
print(' '.join(map(str,avl.preorder())))
    

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![image-20240402221058362](C:\Users\30289\AppData\Roaming\Typora\typora-user-images\image-20240402221058362.png)

大概用时：1h+

### 02524: 宗教信仰

http://cs101.openjudge.cn/practice/02524/



思路：



代码

```python
def init_set(n):
    return list(range(n))
def get_father(x,father):
    if father[x]!=x:
        father[x]=get_father(father[x],father)
    return father[x]
def join(x,y,father):
    fx=get_father(x,father)
    fy=get_father(y,father)
    if fx==fy:
        return
    father[fx]=fy

    return get_father(x,father)==get_father(y,father)

case_num=0
while True:
    n,m=map(int,input().split())
    if n==0 and m==0:
        break
    count=0
    father=init_set(n)
    for _ in range(m):
        s1,s2=map(int,input().split())
        join(s1-1,s2-1,father)
    for i in range(n):
        if father[i]==i:
            count+=1
    case_num+=1
    print(f'Case {case_num}: {count}')


```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![image-20240402222934274](C:\Users\30289\AppData\Roaming\Typora\typora-user-images\image-20240402222934274.png)

大概用时：1h+

## 2. 学习总结和收获

感觉理论知识有点跟不上···前三题还可以，后面三题参考量比较大orz

准备继续多花时间···

