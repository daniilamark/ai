from random import randint

numbers = []
for i in range(10):
    numbers.append(randint(0, 100))

print(numbers)


def sumFor(nums):
    summ = 0
    for num in nums:
        summ += num
    return summ


def sumWhile(nums):
    summ = 0
    i = 1
    while i < len(numbers) + 1:
        summ += nums[i - 1]
        i += 1
    return summ


def sumRecurs(list):
    if not list:
        return 0
    else:
        return list[0] + sumRecurs(list[1:])


def fibonacci(n):
    if n <= 1:
        return n
    else:
        return fibonacci(n - 1) + fibonacci(n - 2)


def fibonacci2(n):
    if n <= 1:
        return n
    else:
        res = fibonacci(n - 1) + fibonacci(n - 2)
        
        return


print(sumFor(numbers))
print(sumWhile(numbers))
print(sumRecurs(numbers))
n = int(input("Введите число: "))
print(fibonacci2(n))

print(numbers)
