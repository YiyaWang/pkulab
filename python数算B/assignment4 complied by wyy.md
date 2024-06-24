# Assignment #4: 排序、栈、队列和树

Updated 0005 GMT+8 March 11, 2024

2024 spring, Complied by ==王诣雅 生命科学学院==





**编程环境**

操作系统：Windows 11 22H2 22621.3155

Python编程环境：Thonny IDE 4.1.4



## 1. 题目

### 05902: 双端队列

http://cs101.openjudge.cn/practice/05902/



思路：

在普通队列上拓展，要注意输出形式。



代码：

```python
t=int(input())
result=[]
for _ in range(t):
    d=[]
    n=int(input())
    for _ in range(n):
        type,ele=map(int,input().split())
        if type==1:
            d.append(ele)
        else:
            if ele==1:
                d.pop()
            else:
                d.pop(0)
                
    if len(d)==0:
        result.append('NULL')
    else:
        new_d=[str(x) for x in d]
        d_str=' '.join(new_d)
        result.append(d_str)
for re in result:
    print(re)

```



代码运行截图 ==（至少包含有"Accepted"）==

![image-20240318183123005](C:\Users\30289\AppData\Roaming\Typora\typora-user-images\image-20240318183123005.png)

大概用时：3分钟



### 02694: 波兰表达式

http://cs101.openjudge.cn/practice/02694/



思路：

使用栈进行模拟。

代码

```python
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
d=list(reversed(d))

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

```



代码运行截图 ==（至少包含有"Accepted"）==

![image-20240318211644484](C:\Users\30289\AppData\Roaming\Typora\typora-user-images\image-20240318211644484.png)

大概用时：5分钟



### 24591: 中序表达式转后序表达式

http://cs101.openjudge.cn/practice/24591/



思路：

遍历中序表达式，这里要尤其注意如何处理小数的输入。如果输入是数字或者小数，放入postfix中，如果是运算符，看看栈里面有无东西，如果没有就直接放入栈，如果有则判断栈顶的运算符与其的优先级，如果栈顶的高，就把栈顶的pop出来放到postfix，反之也直接放入栈。如果遇到左括号，直接放入栈，如果遇到右括号，如果栈里有东西且栈顶不是左括号，把栈顶pop出来放到postfix，反之直接pop栈顶。如果number里面还有数字别忘了放进postfix，最后把栈倒进去，输出postfix即为后序表达式。

代码

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



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![image-20240318223914942](C:\Users\30289\AppData\Roaming\Typora\typora-user-images\image-20240318223914942.png)

大概用时：1小时



### 22068: 合法出栈序列

http://cs101.openjudge.cn/practice/22068/



思路：

对于要判断的字符串，遍历它，对于每一个字符，将原始字符串一个个字符放入栈直到栈顶与该字符一样，pop出栈顶，重复，如果原始字符串用完了但是要判断的字符串还没判断完且发现正在判断的字符与栈顶不同，则NO，如果顺利遍历完要判断的字符串（即每个字符都与栈顶相同且被pop出来），则YES。



代码

```python
def is_valid_sequence(x, seq):
    if len(seq) != len(x):
        return "NO"
    stack = []
    pos = 0  
    for char in seq:
        while pos < len(x) and (not stack or stack[-1] != char):
            stack.append(x[pos])
            pos += 1
        if  stack[-1] != char:
            return "NO"
        stack.pop()      
    return "YES"

x=input()
result=[]
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



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![image-20240319163924320](C:\Users\30289\AppData\Roaming\Typora\typora-user-images\image-20240319163924320.png)

大概用时：1小时



### 06646: 二叉树的深度

http://cs101.openjudge.cn/practice/06646/



思路：

先通过输入构建二叉树节点列表，然后递归遍历求得树的最大深度。

代码

```python
class TreeNode:
    def __init__(self):
        self.left = None
        self.right = None

# Define the function outside of the class, it is a standalone function, not a method
def tree_depth(node):
    if node is None:
        return 0
    left_depth = tree_depth(node.left)
    right_depth = tree_depth(node.right)
    return max(left_depth, right_depth) + 1

n = int(input())  # Number of nodes
tree = [TreeNode() for _ in range(n)]  # List to store all the nodes

# Assign children to each node
for i in range(n):
    L, R = map(int, input().split())
    if L != -1:
        tree[i].left = tree[L-1]  # Assign the left child
    if R != -1:
        tree[i].right = tree[R-1]  # Assign the right child

root = tree[0]  # The first node is the root
depth = tree_depth(root)  # Calculate the depth of the tree
print(depth)  # Print the depth

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![image-20240319210442034](C:\Users\30289\AppData\Roaming\Typora\typora-user-images\image-20240319210442034.png)

大概用时：3小时 orz树还没学懂



### 02299: Ultra-QuickSort

http://cs101.openjudge.cn/practice/02299/



思路：

经典并归排序。

代码

```python
import sys
sys.setrecursionlimit(100000)
d=0
def merge(arr,l,m,r):
    '''对l到m和m到r两段进行合并'''
    global d
    n1=m-l+1#L1长
    n2=r-m#L2长
    L1=arr[l:m+1]
    L2=arr[m+1:r+1]
    ''' L1和L2均为有序序列'''
    i,j,k=0,0,l#i为L1指针，j为L2指针，k为arr指针
    '''双指针法合并序列'''
    while i<n1 and j<n2:
        if L1[i]<=L2[j]:
            arr[k]=L1[i]
            i+=1
        else:
            arr[k]=L2[j]
            d+=(n1-i)#精髓所在
            j+=1
        k+=1
    while i<n1:
        arr[k]=L1[i]
        i+=1
        k+=1
    while j<n2:
        arr[k]=L2[j]
        j+=1
        k+=1
def mergesort(arr,l,r):
    '''对arr的l到r一段进行排序'''
    if l<r:#递归结束条件，很重要
        m=(l+r)//2
        mergesort(arr,l,m)
        mergesort(arr,m+1,r)
        merge(arr,l,m,r)
results=[]
while True:
    n=int(input())#序列长
    if n==0:
        break
    array=[]
    for b in range(n):
        array.append(int(input()))
    d=0
    mergesort(array,0,n-1)
    results.append(d)
for r in results:
    print(r)

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![image-20240319210903581](C:\Users\30289\AppData\Roaming\Typora\typora-user-images\image-20240319210903581.png)

大概用时：2小时



## 2. 学习总结和收获

感觉码力有所提高，但是跟群里的幻神比还是差了太多太多。

决定以后不赶ddl，多抽时间研究。

另：感觉数算挺好玩。python比c要友善好多···

