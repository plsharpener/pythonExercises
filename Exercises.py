"""Python 100例"""


# exercise 1 
"""有四个数字：1、2、3、4，能组成多少个互不相同且无重复数字的三位数？各是多少？"""

def exercise1():
    print("一共有{}种不同的情况".format(4*3*2*1))
    numbers = (1,2,3,4)
    for one in numbers:
        print("one:{}".format(one))
        print(list(numbers))
        print(list(numbers).remove(int(one)))
        for two in list(numbers).remove(one):
            for three in list(numbers).remove(one,two):
                for four in list(numbers).remove(one,two,three):
                    print(1000*one+100*two+10*three+four)

if __name__ == "__main__":
    exercise1()