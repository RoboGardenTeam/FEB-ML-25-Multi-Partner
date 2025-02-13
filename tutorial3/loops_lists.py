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
def Swap(first, second):
    temp = second
    second = first
    first = temp
    return first, second
def Sort(num_list):
    i = 0
    for i in range(len(num_list)):
        for j in range(i, len(num_list)):
            if num_list[i] > num_list[j]:
                num_list[i], num_list[j] = Swap(num_list[i], num_list[j])
    return num_list
            
number_list = [1, 113431, 213, -4, -126, 8, 43, 4231, 1364, 967, 43, 3213, 5314, -4532.3, -513, -52]
print(Sort(number_list))