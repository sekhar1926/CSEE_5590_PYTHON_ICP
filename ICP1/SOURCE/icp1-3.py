str = input("Enter sentence: ")

char = 0
dig = 0

for i in str:
    if i.isdigit():
        dig = dig+1
    elif i.isalpha():
        char = char +1
    else:
        pass

print("no of Letters :",char)
print("no of digits: ", dig)