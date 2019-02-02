n = int(input("Enter n"))

my_list = [int(i) for i in input().split(" ",n-1)]
def avg():
    sum = 0
    for x in range(n):
        sum = sum + my_list[x]
    return (sum/n)
m=avg()
print(m)