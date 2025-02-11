def Average(num_list):
    sum = 0
    for number in num_list:
        sum += number
    average = sum / len(num_list)
    return average
def Maximum(num_list):
    max = num_list[0]
    for number in num_list:
        if max < number:
            max = number
    return max
def Sort(num_list):
    pass
number_list = [1, 113431, 213, -4, -126, 8, 43, 4231, 1364, 967, 3213, 5314, -4532.3, -513, -52]
print(Maximum(number_list))