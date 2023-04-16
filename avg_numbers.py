#!/usr/bin/env python3

import sys

def main():
    numbers = []

    for line in sys.stdin:
        try:
            number = float(line.strip())
            numbers.append(number)
        except ValueError:
            print("Invalid input. Please enter a number or type 'done' to finish.")
    
    average = sum(numbers) / len(numbers) if numbers else 0
    print("Average of the numbers:", average)

if __name__ == "__main__":
    main()