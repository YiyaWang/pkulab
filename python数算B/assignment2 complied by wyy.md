# Assignment #2: 编程练习

Updated 0953 GMT+8 Feb 24, 2024

2024 spring, Complied by ==王诣雅 生命科学学院==



**编程环境**

操作系统：Windows 11 22H2 22621.3155

Python编程环境：Thonny IDE 4.1.4



## 1. 题目

### 27653: Fraction类

http://cs101.openjudge.cn/2024sp_routine/27653/



思路：

​	我觉得这题直接写也可以。于是就按自己的想法先把输入变成四个整数，然后分别表达分子分母，找公因数，然后化简分子分母，最后输出。



##### 代码

```python
list=input().split(' ')
list=[int(z) for z in list]
fenmu=list[1]*list[3]
fenzi=list[0]*list[3]+list[1]*list[2]

if fenzi!=0:
    yinshu_list=[]
    xiaoshu=min(fenzi,fenmu)
    dashu=max(fenzi,fenmu)
    for i in range(1,xiaoshu+1):
        if xiaoshu%i==0:
            yinshu_list.append(i)
            

    gongyinshu_list=[]
    for yinshu in yinshu_list:
        if dashu%yinshu==0:
            gongyinshu_list.append(yinshu)

    mmax=max(gongyinshu_list)
    fenzi=str(int(fenzi/mmax))
    fenmu=str(int(fenmu/mmax))

    print(f'{fenzi}'+'/'+f'{fenmu}')
else:
    fenmu=str(int(fenmu))
    print('0'+'/'+f'{fenmu}')

```



代码运行截图 ==（至少包含有"Accepted"）==

![image-20240307180252479](C:\Users\30289\AppData\Roaming\Typora\typora-user-images\image-20240307180252479.png)



大概用时：1分钟



### 04110: 圣诞老人的礼物-Santa Clau’s Gifts

greedy/dp, http://cs101.openjudge.cn/practice/04110



思路：贪心算法，背包问题，注意按比值排序。



##### 代码

```python
n, maxw = map(int, input().split())
candies = []

for _ in range(n):
    v, w = map(int, input().split())
    candies.append((v, w, v/w))  

candies.sort(key=lambda x: x[2], reverse=True)

sumv = 0
for v, w, _ in candies:
    if maxw >= w:
        maxw -= w
        sumv += v
    else:
        sumv += maxw * (v / w)
        break

print('%.1f' % sumv)

```



代码运行截图 ==（至少包含有"Accepted"）==

![image-20240309221403013](C:\Users\30289\AppData\Roaming\Typora\typora-user-images\image-20240309221403013.png)

大概用时：5分钟

### 18182: 打怪兽

implementation/sortings/data structures, http://cs101.openjudge.cn/practice/18182/



思路：

用字典存储ti与xi，将时刻排序成列表，遍历列表，将每一个t赋给time，将每一时刻前m个技能的血量加起来，用b减去它，若b小于零跳出循环，反之继续遍历列表。结束后根据b给出输出结果。



##### 代码

```python
# ncases = int(input())
for _ in range(ncases):
    n, m, b = map(int, input().split())

    events = {}
    for _ in range(n):
        t, x = map(int, input().split())
        if t in events:
            events[t].append(x)
        else:
            events[t] = [x]

    sorted_times = sorted(events.keys())  # Get the sorted times

    time = 0
    for t in sorted_times:
        # Skip to the next relevant time if the current time is not yet reached
        if t > time:
            time = t

        # Process events at this time
        now_x_list = sorted(events[t], reverse=True)
        total_x = sum(now_x_list[:m])  # Only consider up to m largest x values

        b -= total_x
        if b <= 0:
            break  # b has been satisfied

    if b > 0:
        print('alive')
    else:
        print(time)


```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![image-20240307224126379](C:\Users\30289\AppData\Roaming\Typora\typora-user-images\image-20240307224126379.png)

大概用时：10分钟（老是超时<angry>)

感想：有点刷新我对字典的认知以及教会我要智慧地选择遍历条件（最好不要常数一步步爬···）





### 230B. T-primes

binary search/implementation/math/number theory, 1300, http://codeforces.com/problemset/problem/230/B



思路：

使用艾氏筛打表。



##### 代码

```python
import math

def sieve_of_eratosthenes(max_num):
    is_prime = [True] * (max_num + 1)
    primes = []
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.sqrt(max_num)) + 1):
        if is_prime[i]:
            primes.append(i)
            for j in range(i*i, max_num + 1, i):
                is_prime[j] = False
    for i in range(int(math.sqrt(max_num)) + 1, max_num + 1):
        if is_prime[i]:
            primes.append(i)
    return primes

def is_sqrt(x):
    root = int(math.sqrt(x))
    return root * root == x

def is_T_prime(x, primes_set):
    if x == 1:
        return False
    if is_sqrt(x) and int(math.sqrt(x)) in primes_set:
        return True
    return False

n = int(input())
num_list = map(int, input().split())

primes = sieve_of_eratosthenes(1000000)  # Precompute primes up to 10^6
primes_set = set(primes)  # Convert list to set for O(1) lookup

for num in num_list:
    if is_T_prime(num, primes_set):
        print('YES')
    else:
        print('NO')

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![image-20240309140201858](C:\Users\30289\AppData\Roaming\Typora\typora-user-images\image-20240309140201858.png)

大概用时：20分钟



### 1364A. XXXXX

brute force/data structures/number theory/two pointers, 1200, https://codeforces.com/problemset/problem/1364/A



思路：数论分析，发现最优解就是从左或右找到第一个不是x倍数的数，然后删去端侧数列。比较输出最长子序列。



##### 代码

```python
t = int(input()) 
for _ in range(t):
    n, x = map(int, input().split())
    arr = list(map(int, input().split()))
    total_sum = sum(arr)
    
    if total_sum % x != 0:
        print(n)
    else:
        prefix_sum = 0
        min_length = n + 1  
        for i in range(n):
            prefix_sum += arr[i]
            if prefix_sum % x != 0:
                min_length = min(min_length, i + 1)
        
        suffix_sum = 0
        for i in range(n-1, -1, -1):
            suffix_sum += arr[i]
            if suffix_sum % x != 0:
                min_length = min(min_length, n - i)
        
        # 计算最长子数组的长度
        if min_length == n + 1:
            print(-1)  # 找不到符合条件的子数组
        else:
            print(n - min_length)  # 输出最长子数组长度

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![image-20240309162051418](C:\Users\30289\AppData\Roaming\Typora\typora-user-images\image-20240309162051418.png)

大概用时：20分钟



### 18176: 2050年成绩计算

http://cs101.openjudge.cn/practice/18176/



思路：在上面的T_prime的基础上完成此题。



##### 代码

```python
import math

def sieve_of_eratosthenes(max_num):
    is_prime = [True] * (max_num + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.sqrt(max_num)) + 1):
        if is_prime[i]:
            for j in range(i*i, max_num + 1, i):
                is_prime[j] = False
    return is_prime

def is_sqrt(x):
    root = int(math.sqrt(x))
    return root * root == x

def is_T_prime(x, primes_set):
    if x == 1:
        return False
    if is_sqrt(x) and primes_set[int(math.sqrt(x))]:
        return True
    return False

primes_set = sieve_of_eratosthenes(10000)

stu_num,class_num=map(int,input().split(' '))

for stu in range(stu_num):
    grade_list=list(map(int,input().split(' ')))  
    T_prime_grade=[]
    for grade in grade_list:
        if is_T_prime(grade,primes_set):
            T_prime_grade.append(grade)
    if sum(T_prime_grade)==0:
        print(0)
    else:
        result=sum(T_prime_grade)/len(grade_list)
        print('%.2f'%result) 

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![image-20240309202553626](C:\Users\30289\AppData\Roaming\Typora\typora-user-images\image-20240309202553626.png)

大概用时：5分钟



## 2. 学习总结和收获

​	非常伤心，第一遍的作业忘记保存也忘记提交了，于是又写了一遍。很高兴的是，写第二遍的时候顺畅多了，发现还是要多练，而且不排除重复练习。

​	逐渐C转py上路了···

​	继续加油吧！





