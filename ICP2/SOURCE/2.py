stack = []
while True:
    print("1.push 2.pop 3.exit")
    x=int(input("Enter option"))
    if x==1:
        y =(input("enter element to push"))
        stack.append(y)
        print(stack)
    elif x==2:
        stack.pop()
        print(stack)
    else:
        exit()



