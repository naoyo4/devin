"""
A simple CLI calculator program.
Usage:
    python SimpleCalculator.py add 5 3
    python SimpleCalculator.py subtract 10 4
    python SimpleCalculator.py multiply 6 7
    python SimpleCalculator.py divide 20 5
"""

import sys


def add(a, b):
    """Add two numbers and return the result."""
    return a + b


def subtract(a, b):
    """Subtract b from a and return the result."""
    return a - b


def multiply(a, b):
    """Multiply two numbers and return the result."""
    return a * b


def divide(a, b):
    """Divide a by b and return the result."""
    if b == 0:
        return "Error: Division by zero"
    return a / b


def main():
    """Main function to parse arguments and perform calculations."""
    if len(sys.argv) < 4:
        print("Usage: python SimpleCalculator.py <operation> <number1> <number2>")
        print("Operations: add, subtract, multiply, divide")
        return

    operation = sys.argv[1].lower()
    
    try:
        num1 = float(sys.argv[2])
        num2 = float(sys.argv[3])
    except ValueError:
        print("Error: Please provide valid numbers")
        return

    result = None
    
    if operation == "add":
        result = add(num1, num2)
    elif operation == "subtract":
        result = subtract(num1, num2)
    elif operation == "multiply":
        result = multiply(num1, num2)
    elif operation == "divide":
        result = divide(num1, num2)
    else:
        print("Error: Unknown operation")
        print("Operations: add, subtract, multiply, divide")
        return

    print(f"Result: {result}")


if __name__ == "__main__":
    main()
