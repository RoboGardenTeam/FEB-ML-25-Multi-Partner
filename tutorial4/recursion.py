def Factorial(num):
    if num == 1:
        return 1
    return num * Factorial(num - 1)
def Fibonacci(loc):
    if loc <= 1:
        return 1
    return Fibonacci(loc - 1) + Fibonacci(loc - 2)

number = 5
print(Fibonacci(number))

#4 * Factorial(3)
#3 * Factorial(2)
#2 * Factorial(1)
#1
# 1 2 3 5 8 13

#5! = 5 x 4!
#4! = 4 x 3!


