from collections import deque
queue = deque(["1","2","3","4","5"])
while True:
    print("1.enqueue 2.dequeue 3.exit")
    x=int(input("Enter option"))
    if x==1:
        y =(input("enter element"))
        queue.append(y)
        print(queue)
    elif x==2:
        popped=queue.popleft()
        print("dequeued element is ",popped)
        print(queue)
    else:
        exit()
