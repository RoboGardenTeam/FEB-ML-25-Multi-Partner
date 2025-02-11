def Swap(first, second):
    temp = second
    second = first
    first = temp
    return first, second

first_number = 78
second_number = 89
print(first_number, second_number)
first_swapped, second_swapped = Swap(first_number, second_number)
print(first_swapped, second_swapped)