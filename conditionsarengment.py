import csv
from itertools import product

def generate_permutations_with_repetition(numbers, length, output_file):

    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = tuple(f'column_{i}' for i in range(length))
        writer = csv.writer(csvfile)
        writer.writerow(fieldnames)

        for permutation in product(numbers, repeat=length):
            writer.writerow(permutation)

# لیست اعداد مورد نظر
numbers = [1, 2, 3, 4, 5]
# طول هر آرایش
length = 6
# نام فایل خروجی
output_file = 'permutations_with_repetition.csv'

generate_permutations_with_repetition(numbers, length, output_file)