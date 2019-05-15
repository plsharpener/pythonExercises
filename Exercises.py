"""Python 100例"""
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



if __name__ == "__main__":
    #print(exercise6(100))
    # print(exercise11(11))
    exercise12()