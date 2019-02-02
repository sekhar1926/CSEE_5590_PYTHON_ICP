fname = input('First Name: ')
lname = input('Last Name: ')

def reverse(str):
    revstr = ""
    for i in str:
        revstr = i + revstr
    return revstr

print(reverse(fname)+" "+reverse(lname))