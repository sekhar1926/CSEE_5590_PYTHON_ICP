class Employee:
    count = 0
    total_salary = 0



    def __init__(self, name, family, salary, dept):
        self.name = name
        self.family = family
        self.salary = salary
        self.dept = dept
        Employee.count += 1
        Employee.total_salary += salary

    def emp_count(self):

        print("total number of employees", Employee.count)

    def avg_salary(self):

        avg_sal = Employee.total_salary / Employee.count
        print("average salary:", avg_sal)

    def sample_func(self):
        print('calling base class member function')


class Full_time_employee(Employee):
    def __init__(self):
        print('Full time employee(sub class)')


n = int(input("No of employees:"))
for i in range(n):
    nam= input("name:")
    fam = input("family:")
    sal = float(input("salary:"))
    dep = input("dept:")
    e = Employee(nam,fam,sal,dep)
c = Full_time_employee()
c.emp_count()
c.avg_salary()
e.sample_func()
