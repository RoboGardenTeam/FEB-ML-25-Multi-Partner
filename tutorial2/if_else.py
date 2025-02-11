def Calculate(first, second, op):
    if op == '+':
        return first + second
    elif op == '-':
        return first - second
    elif op == '*':
        return first * second
    elif op == '/':
        if second != 0:
            return int(first) / int(second)
        else:
            return "Cannot divide by Zero"
    else:
        return 'Please enter a valid operation'
    


first_number = float(input('Please enter the first number: '))
second_number = float(input('Please enter the second number: '))
operation = input('Please enter an operation: ')
result = Calculate(first_number, second_number, operation)
print(result)