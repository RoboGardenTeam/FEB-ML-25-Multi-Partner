def Calculate(first, second, op):
   match op:
       case '+':
           return first + second
       case '-':
           return first - second
       case '*':
           return first * second
       case '/':
           if second != 0:
            return first / second
           else:
               return 'division by zero error'
       case _:
           return 'Something went wrong'
    
first_number = float(input('Please enter the first number: '))
second_number = float(input('Please enter the second number: '))
operation = input('Please enter an operation: ')
while operation not in ['+', '-', '/', '*']:
    operation = input('Please enter a valid operation: ')
result = Calculate(first_number, second_number, operation)
print(result)