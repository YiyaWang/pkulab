## Assignment #1

###  Updated 0940 GMT+8 Feb 21, 2024

### 2024 spring, Complied by 王诣雅 生命科学学院



编程环境：

操作系统：Windows 11 22H2 22621.3155

Python编程环境：Thonny IDE 4.1.4



1.题目

20742: 泰波拿契數

思路：

先根据泰波拿契數的特点利用递归思想定义函数，再运行函数。

代码：

def T(n):
    if n==0:
        return 0
    elif n==1 or n==2:
        return 1
    else:
        return T(n-1)+T(n-2)+T(n-3)
    return T(n)
n=int(input())
print(T(n))

代码运行截图：

![image-20240221102605487](C:\Users\30289\AppData\Roaming\Typora\typora-user-images\image-20240221102605487.png)

大概用时：1分钟



58A. Chat room

思路：

定义函数，以输入的需检测的字符串为变量，对变量字符串的每一个字符进行遍历，从前往后核验是否与"hello"中字符对应相同。若均找到，则YES；反之，则NO。

代码：

def can_say_hello(s):
    hello="hello"
    hello_index=0
    for char in s:
        if char==hello[hello_index]:
            hello_index+=1
            if hello_index==len(hello):
                return "YES"
    return "NO"
s=input()
print(can_say_hello(s))

代码运行截图：

![image-20240221141558825](C:\Users\30289\AppData\Roaming\Typora\typora-user-images\image-20240221141558825.png)

大概用时：3分钟



118A. String Task

思路:

遍历输入的字符串，用set进行成员资格测试，把辅音选择出来并前面加点放入列表，最后用join把列表元素整合成一个字符串输出。

代码：

def changed_string(s):
    vowels=set("AOYEUIaoyeui")
    result=[]
    for char in s:
        if char not in vowels:
            result.append("."+char.lower())
    return "".join(result)
s=input()
print(changed_string(s))

代码运行截图：

![image-20240221151722299](C:\Users\30289\AppData\Roaming\Typora\typora-user-images\image-20240221151722299.png)

大概用时：3分钟



22359: Goldbach Conjecture

思路：

定义判断质数函数和寻找两个质数因子的函数进行求解。

代码：

def is_prime(num):
    if num < 2:
        return False
    for i in range(2, int(num**0.5) + 1):
        if num % i == 0:
            return False
    return True

def find_primes(n):
    for j in range(2, n):
        if is_prime(j) and is_prime(n - j):
            return f"{j} {n - j}"  

n = int(input())  
print(find_primes(n))

代码运行截图：

![image-20240221154308881](C:\Users\30289\AppData\Roaming\Typora\typora-user-images\image-20240221154308881.png)

大概用时：4分钟



23563: 多项式时间复杂度

思路：

定义函数：把表达式用split函数分割成项，再把项分割出系数和指数。如果该项没有n，则无法被分割,max-exponent=0。如果n无指数（或者说指数为1），max_exponent=1。如果n有指数且系数不为0，比较记录max_exponent。最后分情况输出。运行函数。

代码：

def find_time_complexity(expression):
    terms=expression.split("+")
    max_exponent=-1
    for term in terms:
        parts=term.split("n^")
        if len(parts)==2:
            if parts[0]!="0":
                exponent=int(term.split("^")[1])
                max_exponent=max(exponent,max_exponent)
        elif "n" in term and "^" not in term:
            max_exponent=max(1,max_exponent)
    if max_exponent==-1:
        return "n^0"
    else:
        return f"n^{max_exponent}"
    
expression=input()
print(find_time_complexity(expression))

​        

代码运行截图：

![image-20240221165221947](C:\Users\30289\AppData\Roaming\Typora\typora-user-images\image-20240221165221947.png)

大概用时：10分钟



24684: 直播计票

思路：

运用字典和集合，将输入转化为列表，遍历列表中的元素，把元素放入集合并且字典中相应的值设为1，如果已经放入，再次出现时值加一。创建列表：出现次数最多的数；最多出现次数初始化为0；用字典的键判断是不是出现次数最多的数，如果是则置换列表或者加入列表。最后字符串形式输出列表。

代码：

#将输入的字符串转化为列表
nums=[int(z) for z in input().split()]
#创建集合和字典
s=set()
dic=dict()
#把列表中的元素放进集合和字典，如果已经加入集合，当列表中再次出现该数时，字典的值加一
for num in nums:
    if num in s:
        dic[num]+=1
    else:
        s.add(num)
        dic[num]=1
#创建列表：出现最多次数的数;最多出现次数归零
max_count_num=[]
max_count=0
#判断是不是出现次数最多的数，如果是的话把值放在新建立的列表里
for number,count in dic.items():
    if count>max_count:
        max_count_num=[number]
        max_count=count
    elif count==max_count:
        max_count_num.append(number)
print(" ".join(map(str,sorted(max_count_num))))        

代码运行截图：

![image-20240222144949542](C:\Users\30289\AppData\Roaming\Typora\typora-user-images\image-20240222144949542.png)

大概用时：10分钟



2.学习总结和收获

我计算概率B学的是C，通过python练习迅速掌握python基础语法和基础算法。

本次作业考察了递归、字符串、集合、基本函数（如判断素数等）、字典、判断、条件循环等基础结构等等，使我对python的基础语法和基础算法有了较大的巩固和提升。