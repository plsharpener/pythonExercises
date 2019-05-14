"""Python 100例"""


# exercise 1 
"""有四个数字：1、2、3、4，能组成多少个互不相同且无重复数字的三位数？各是多少？"""

def exercise1():
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

if __name__ == "__main__":
    exercise1()