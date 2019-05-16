"""Python 100例 https://www.runoob.com/python/python-100-examples.html"""
import sys
# exercise 1 
def exercise1() -> None:
    """有四个数字：1、2、3、4，能组成多少个互不相同且无重复数字的三位数？各是多少？"""
    numbers = (1,2,3,4,5,6)
    for one in numbers:
        for two in numbers:
            if two in [one]:
                continue
            else:
                for three in numbers:
                    if three in [one,two]:
                        continue
                    else:
                        for four in numbers:
                            if four in [one,two,three]:
                                continue
                            else:
                                print(1000*one+100*two+10*three+four)
    print("一共有{}种不同的情况".format(len(numbers)*(len(numbers)-1)*(len(numbers)-2)*(len(numbers)-3)))


# exercise 2
def exercise2() -> None:
    """企业发放的奖金根据利润提成。利润(I)低于或等于10万元时，奖金可提10%；
   利润高于10万元，低于20万元时，低于10万元的部分按10%提成，高于10万元的部分，
   可提成7.5%；20万到40万之间时，高于20万元的部分，可提成5%；40万到60万之间时
   0万元的部分，可提成3%；60万到100万之间时，高于60万元的部分，可提成1.5%，
   高于100万元时，超过100万元的部分按1%提成，从键盘输入当月利润I，求应发放奖金总数？
    """
    I = float(input("输入本月收益："))
    jiangjin = 0.0
    #方法1
    # if I <= 10:
    #     jiangjin = I*10*0.01
    # elif I <= 20:
    #     jiangjin = 1+(I-10)*7.5*0.01
    # elif I <= 40:
    #     jiangjin = 1+0.75+(I-20)*5*0.01
    # elif I <= 60:
    #     jiangjin = 1+0.75+1+(I-40)*3*0.01
    # elif I <= 100:
    #     jiangjin = 1+0.75+1+0.6+(I-60)*1.5*0.01
    # else:
    #     jiangjin = 1+0.75+1+0.6+0.6+(I-100)*1*0.01
    #方法2
    limit = [100,60,40,20,10,0]
    ratio = [0.01,0.015,0.03,0.05,0.075,0.1]
    for i in range(6):
        if I > limit[i]:
            jiangjin += (I-limit[i])*ratio[i]
            I= limit[i]
    print("总奖金是{}万元".format(jiangjin))

# exercise 3
def exercise3() -> None:
    """一个整数，它加上100后是一个完全平方数，再加上168又是一个完全平方数，请问该数是多少？"""
    num=0
    while(1):
        if not((num+100)**0.5%1 or (num+168)**0.5%1):
            print("这个整数是：{}".format(num))
            sys.exit()
        num+=1

# exercise 4
def exercise4() -> None:
    """输入某年某月某日，判断这一天是这一年的第几天？"""
    date = input("输入日期xxxx-xx-xx：")
    year,month,day = tuple(date.split("-"))
    year = int(year)
    month = int(month)
    print(month)
    day = int(day)
    total = int(0)
    months = [31,28,31,30,31,30,31,31,30,31,30,31]

    for i in range(1,month):
        total += months[i-1]
        if i == 2:
            if year%100:
                if not(year%4):
                    total+=1
            else:
                if not(year%400):
                    total+=1
    total += day
    print(total)

# exercise 5
def exercise5() -> None:
    """输入三个整数x,y,z，请把这三个数由小到大输出"""
    number = input("输入三个数字")
    x,y,z = tuple(number.split(","))
    x = int(x)
    y = int(y)
    z = int(z)
    if x>y:
        x,y=y,x
    if x>z:
        x,z=z,x
    if y>z:
        y,z=z,y
    print(x,y,z)

# exercise 6
def exercise6(n:int) -> int:
    """斐波那契数列"""
    if n==0:
        return 0
    if n==1:
        return 1
    return exercise6(n-1)+exercise6(n-2)

# exercise 7
def exercise7() -> None:
    """复制列表"""
    l1 = [1,2,3,4,5,6,7,8,9,10]
    l2 = l1[:]
    # l2 = l1
    l1.pop()
    print("L1:",l1)
    print("L2:",l2)

# exercise 8
def exercise8() -> None:
    """输出 9*9 乘法口诀表"""
    for i in range(1,10):
        for j in range(1,i+1):
            print("{}*{}={}".format(i,j,i*j),end=" ")
        print("\n")

import time
# exercise 9
def exercise9() -> None:
    """暂停一秒输出"""
    time.sleep(1)
    print("sleep 1s")

# exercise 10
def exercise10() -> None:
    """暂停一秒输出，并格式化当前时间"""
    print(time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time.time())))
    time.sleep(1)
    print(time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time.time())))

# exercise 11
def exercise11(n:int) -> None:
    """古典问题：有一对兔子，从出生后第3个月起每个月都生一对兔子，小兔子长到第三个月后每个月又生一对兔子，假如兔子都不死，问每个月的兔子总数为多少？"""
    if n == 1 or n == 2:
        return 1
    return exercise11(n-1)+exercise11(n-2)

# exercise 12
def exercise12() -> None:
    """判断101-200之间有多少个素数，并输出所有素数。"""
    countPrimeNumber = int(0)
    for i in range(101,201):
        isPrimeNumber = True
        for j in range(2,int(1+i/2)):
            isPrimeNumber = isPrimeNumber and (i%j)
        if isPrimeNumber:
            print(i)
            countPrimeNumber+=1
            if not(countPrimeNumber%10):
                print(" ")
    print("The total is {}".format(countPrimeNumber))

# exercise 13
def exercise13() -> None:
    """打印出所有的"水仙花数"，所谓"水仙花数"是指一个三位数，其各位数字立方和等于该数本身。例如：153是一个"水仙花数"，因为153=1的三次方＋5的三次方＋3的三次方"""
    for i in range(100,1000):
        stri = str(i)
        # print(stri)
        if (eval(stri[0])**3+eval(stri[1])**3+eval(stri[2])**3) == i:   #参考https://www.cnblogs.com/wuxiangli/p/6046800.html
            print(i)                                                    #eval(string) 字符串转化为算式   repr(x)  将对象x转化成表达字符串 

# exercise 14
def PrimeNumber(n:int) -> list:
    """判断1-n之间有多少个素数，并输出所有素数列表。"""
    PrimeNumbers = list()
    for i in range(1,n+1):
        isPrimeNumber = True
        for j in range(2,int(1+i/2)):
            isPrimeNumber = isPrimeNumber and (i%j)
        if isPrimeNumber:
            PrimeNumbers.append(i)
    return PrimeNumbers

def exercise14(n:int) -> list:
    """将一个正整数分解质因数。例如：输入90,打印出90=2*3*3*5"""
    PrimeNumbers = PrimeNumber(n)
    # print(PrimeNumbers)
    factors = list()
    i = int(1)
    # print("{}=".format(n),end="")
    while(n!=1):
        if not(n%PrimeNumbers[i]):
            factors.append(PrimeNumbers[i])
            n = n/PrimeNumbers[i]
        else:
            i+=1
    # print(factors[0],end="")
    # for i in factors[1:]:
    #     print("*{}".format(i),end="")
    return factors

# exercise 15
def exercise15() -> None:
    """利用条件运算符的嵌套来完成此题：学习成绩>=90分的同学用A表示，60-89分之间的用B表示，60分以下的用C表示。"""
    score = int(input("输入成绩："))
    if score >=90:
        print("A")
    elif score >=60:
        print("B")
    else:
        print("C")

# exercise 16
import datetime
def exercise16() -> None:
    """输出指定格式的日期"""
    # 输出今日日期，格式是dd/mm/yyyy
    print(datetime.date.today().strftime('%d/%m/%Y'))

    # 创建时间对象
    miyazakiBirthDate = datetime.date(1941,1,5)
    print(miyazakiBirthDate.strftime('%d/%m/%Y'))

    # 日期算术运算
    miyazakiBirthNextDay = miyazakiBirthDate + datetime.timedelta(days=1)
    print(miyazakiBirthNextDay.strftime("%d/%m/%Y"))

    # 日期替换
    miyazakiFirstBirthday = miyazakiBirthDate.replace(year=miyazakiBirthDate.year+1)
    print(miyazakiFirstBirthday.strftime('%d/%m/%Y'))

# exercise 17
def exercise17() -> None:
    """输入一行字符，分别统计出其中英文字母、空格、数字和其它字符的个数。"""
    latters = int(0)
    kongge = int(0)
    digit = int(0)
    others = int(0)
    strings = input("输入一串字符：")
    for s in strings:
        if s.isdigit():
            digit +=1
        elif s.isalpha():
            latters+=1
        elif s == " ":
            kongge +=1
        else:
            others +=1
    print("latters = {} digit = {} kongge = {} others = {}".format(latters,digit,kongge,others))

# exercise 18
def exercise18() -> None:
    """求s=a+aa+aaa+aaaa+aa...a的值，其中a是一个数字。例如2+22+222+2222+22222(此时共有5个数相加)，几个数相加由键盘控制"""
    counts = int(input("n="))
    a = input("a=")
    total = int(0)
    for i in range(1,counts+1):
        print(a*i)
        total += eval(a*i)
    print(total)

# exercise 19
def exercise19() -> list:
    """一个数如果恰好等于它的因子之和，这个数就称为"完数"。例如6=1＋2＋3.编程找出1000以内的所有完数。"""
    wanshu = list()
    for number in range(2,1000):
        factors = list()
        for i in range(1,number):
            if not(number%i):
                factors.append(i)
        # print(factors)
        he = int(0)
        for i in factors:
            he+=i
        if number == he:
            wanshu.append(number)
            print(number)
    return wanshu

# exercise 20
def exercise20() -> None:
    """一球从100米高度自由落下，每次落地后反跳回原高度的一半；再落下，求它在第10次落地时，共经过多少米？第10次反弹多高？"""
    print("第十次反弹高度：{}米".format(100/(2**10)))
    total = 100
    for i in range(1,10):
        total += 2*100/(2**i)
    print(total)

# exercise 21
def exercise21(n:int) -> int:
    """猴子吃桃问题：猴子第一天摘下若干个桃子，当即吃了一半，还不瘾，又多吃了一个第二天早上又将剩下的桃子吃掉一半，
    又多吃了一个。以后每天早上都吃了前一天剩下的一半零一个。到第10天早上想再吃时，见只剩下一个桃子了。求第一天共摘了多少。"""
    if n == 1:
        return 1
    return 2*(exercise21(n-1)+1)

# exercise 22
def exercise22() -> None:
    """两个乒乓球队进行比赛，各出三人。甲队为a,b,c三人，乙队为x,y,z三人。已抽签决定比赛名单。有人向队员打听比赛的名单。a说他不和x比，c说他不和x,z比，请编程序找出三队赛手的名单。"""
    pass

# exercise 23
def exercise23(n:int) -> None:
    """打印出如下图案（菱形）"""
    if not(n%2):
        n+=1
    for i in range(n):
        if (i <= int(n/2)):
            print((((2*i)+1)*"*").center(n," "))
        else:
            print((((2*(n-i-1))+1)*"*").center(n," "))
    
# exercise 24
def exercise24() -> None:
    """有一分数序列：2/1，3/2，5/3，8/5，13/8，21/13...求出这个数列的前20项之和"""
    total = 0.0
    fenzi = 2.0
    fenmu = 1.0
    for _ in range(20):
        total += fenzi/fenmu
        fenmu,fenzi = fenzi,fenzi+fenmu
    print(total)
        





if __name__ == "__main__":
    #print(exercise6(100))
    # print(exercise11(11))
    # print(PrimeNumber(100))
    # print(exercise14(28))
    exercise24()