"""Python 100例"""




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
    if I <= 10:
        jiangjin = I*10*0.01
    elif I <= 20:
        jiangjin = 1+(I-10)*7.5*0.01
    elif I <= 40:
        jiangjin = 1+0.75+(I-20)*5*0.01
    elif I <= 60:
        jiangjin = 1+0.75+1+(I-40)*3*0.01
    elif I <= 100:
        jiangjin = 1+0.75+1+0.6+(I-60)*1.5*0.01
    else:
        jiangjin = 1+0.75+1+0.6+0.6+(I-100)*1*0.01
    print("总奖金是{}万元".format(jiangjin))
if __name__ == "__main__":
    exercise2()