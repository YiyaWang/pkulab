# Assignment #3: March月考

Updated 1537 GMT+8 March 6, 2024

2024 spring, Complied by ==王诣雅 生命科学学院==



**编程环境**

编程环境：

操作系统：Windows 11 22H2 22621.3155

Python编程环境：Thonny IDE 4.1.4





## 1. 题目

**02945: 拦截导弹**

http://cs101.openjudge.cn/practice/02945/



思路：

寻找最长非递增子序列。使用dp。

##### 代码

```python
def max_intercept_missiles(k, heights):
    dp = [1] * k  
    
    for i in range(1, k):
        for j in range(i):
            if heights[i] <= heights[j]:
                dp[i] = max(dp[i], dp[j] + 1)
                
    return max(dp)  

k = int(input())
heights=list(map(int,input().split()))


print(max_intercept_missiles(k, heights))

```



代码运行截图 ==（至少包含有"Accepted"）==

![image-20240310153108828](C:\Users\30289\AppData\Roaming\Typora\typora-user-images\image-20240310153108828.png)

大概用时：10分钟





**04147:汉诺塔问题(Tower of Hanoi)**

http://cs101.openjudge.cn/practice/04147



思路：

递归



##### 代码

```python
def hannuota(k,source,target,auxiliary):
    if k==1:
        print(f'1:{source}->{target}')
        return
    hannuota(k-1,source,auxiliary,target)
    print(f'{k}:{source}->{target}')
    hannuota(k-1,auxiliary,target,source)
    
k,source,auxiliary,target=input().split()
k=int(k)
hannuota(k,source,target,auxiliary)

```



代码运行截图 ==（至少包含有"Accepted"）==

![image-20240310155116508](C:\Users\30289\AppData\Roaming\Typora\typora-user-images\image-20240310155116508.png)

大概用时：5分钟



**03253: 约瑟夫问题No.2**

http://cs101.openjudge.cn/practice/03253



思路：

先将小孩按编号存在一个队列里，设置一个index，初始化为p-1，进行索引，每次报数m，则每次index更新为(index+m-1)%len(children),然后将children队列里的小孩pop到新列表里，直到小孩全都出去。按顺序输出小孩的编号即可。



##### 代码

```python
def josephus_problem_2(n, p, m):
    children = list(range(1, n + 1))
    out_children = []
    index = p - 1  # Adjust the starting index based on p

    while children:
        index = (index + m - 1) % len(children)  # Find the next child to remove
        out_children.append(children.pop(index))  # Remove the child

    return out_children

# Loop to handle multiple sets of inputs
while True:
    n, p, m = map(int, input().split())
    if n == 0 and p == 0 and m == 0:  # Termination condition
        break
    # Get the order in which children are removed
    new_list = josephus_problem_2(n, p, m)
    # Print the order, formatted as required
    print(','.join(map(str, new_list)))

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![image-20240310195155280](C:\Users\30289\AppData\Roaming\Typora\typora-user-images\image-20240310195155280.png)

大概用时：30分钟



**21554:排队做实验 (greedy)v0.2**

http://cs101.openjudge.cn/practice/21554



思路：贪心算法



##### 代码

```python
num_of_stu=int(input())
time_list=list(map(int,input().split()))
new_time_list=sorted(time_list)

dic=dict()
x=1
for time in time_list:
    if time in dic:
        dic[time].append(x)
    else:
        dic[time]=[x]
    x+=1

new_dic=sorted(dic)

b=[]

for key in new_dic:
    for value in dic[key]:
        b.append(value)
print(' '.join(map(str,b)))

y=num_of_stu-1
wtime=0
for key in new_time_list:
    wtime+=key*y
    y-=1

awtime=wtime/num_of_stu
print('%.2f'%awtime) 

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![image-20240312161544135](C:\Users\30289\AppData\Roaming\Typora\typora-user-images\image-20240312161544135.png)

大概用时：10分钟



**19963:买学区房**

http://cs101.openjudge.cn/practice/19963



思路：分别排序距离和价格，分别找到符合要求的学区房，求并集。



##### 代码

```python
def find_median(values):
    sorted_values = sorted(values)
    n = len(sorted_values)
    
    
    if n % 2 == 0:
        return (sorted_values[n//2 - 1] + sorted_values[n//2]) / 2
    else:
        return sorted_values[n//2]

n = int(input().strip())
pairs = input().strip().split()
prices = list(map(int, input().strip().split()))

distances = [sum(map(int, i[1:-1].split(','))) for i in pairs]
cost_efficiency_ratios = [distances[i] / prices[i] for i in range(n)]

median_ratio = find_median(cost_efficiency_ratios)
median_price = find_median(prices)

worthy_houses = 0
for i in range(n):
    if cost_efficiency_ratios[i] > median_ratio and prices[i] < median_price:
        worthy_houses += 1

print(worthy_houses)

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![image-20240312184448021](C:\Users\30289\AppData\Roaming\Typora\typora-user-images\image-20240312184448021.png)

大概用时：60分钟+（debug de不出来了···借助了一点外援···）

```python

#以下是我原来的代码，样例都ac了，但是提交是wronganswer，实在看不出来为什么qwq
#另：确实写的太丑了···

import ast
num_of_house = int(input())
tup_strings = input().split()
distancelist = [ast.literal_eval(t) for t in tup_strings]
moneylst = list(map(int, input().split()))

length=[]
n=0
for tup in distancelist:
    length.append(sum(tup)/moneylst[n])
    n+=1


want_1=[]
want_2=[]

new_moneylst=sorted(moneylst)

tmp_1 = [i for i in new_moneylst if len(new_moneylst)%2!=0 and i < new_moneylst[(len(new_moneylst)-1)//2]]
tmp_2 = [i for i in new_moneylst if len(new_moneylst)%2==0 and i < (new_moneylst[len(new_moneylst)//2]+new_moneylst[len(new_moneylst)//2-1])/2]
if len(new_moneylst)%2!=0 :
    for j in tmp_1:
        summ=0
        for ele in moneylst:
            summ+=1
            if ele==j:
                want_1.append(summ)
else:
    for j in tmp_2:
        summ=0
        for ele in moneylst:
            summ+=1
            if ele==j:
                want_1.append(summ)

new_length=sorted(length)

tmpt_1 = [l for l in new_length if len(new_length)%2!=0 and l > new_length[(len(new_length)-1)//2]]
tmpt_2 = [l for l in new_length if len(new_length)%2==0 and l > (new_length[len(new_length)//2]+new_length[len(new_length)//2-1])/2]
if len(new_length)%2!=0 :
    for k in tmpt_1:
        summ=0
        for ele in length:
            summ+=1
            if ele==k:
                want_2.append(summ)
else:
    for k in tmpt_2:
        summ=0
        for ele in length:
            summ+=1
            if ele==k:
                want_2.append(summ)

num=0
for house in want_1:
    if house in want_2:
        num+=1
print(num)


#？？？
```



**27300: 模型整理**

http://cs101.openjudge.cn/practice/27300



思路：

将模型名和参数分开存储，参数保留原格式同时转换为统一单位用于排序。最后，根据模型名字典序和参数量大小输出。

##### 代码

```python
n = int(input())  # Number of models
models = {}  # Dictionary to hold model names and their parameters

# Function to convert parameters to a uniform representation for sorting
def param_key(param):
    value, unit = float(param[:-1]), param[-1]
    return value * 1000 if unit == 'B' else value

# Read each model, split into name and parameters, and store in the dictionary
for _ in range(n):
    model_str = input()
    name, param = model_str.split('-')
    if name not in models:
        models[name] = []
    models[name].append(param)

# Sort the parameters for each model using the converted values for comparison
for name in models.keys():
    models[name].sort(key=param_key)

# Sort the model names and print the results
for name in sorted(models.keys()):
    formatted_params = ', '.join(models[name])
    print(f"{name}: {formatted_params}")

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![image-20240312201633970](C:\Users\30289\AppData\Roaming\Typora\typora-user-images\image-20240312201633970.png)

大概用时：40分钟



## 2. 学习总结和收获

感觉贪心掌握得比较好。尝试用正则表达式但是寄了。有一些具体的语法记得不牢，导致细节处还需加力。

加油加油！

感觉自己的很多程序都写得有点丑···希望数据结构掌握后能优美起来！

