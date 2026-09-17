"""
Stack - Problem Template

Problem:
    <Add problem statement here>

Approach:
    <Describe your approach>

Time Complexity:  O(?)
Space Complexity: O(?)
"""


# Creating a stack
def create_stack():
    stack = []
    return stack


# Creating an empty stack
def check_empty(stack):
    return len(stack) == 0


# Adding items into the stack
def push(stack, item):
    stack.append(item)
    print("pushed item: " + item)


# Removing an element from the stack
def pop(stack):
    if (check_empty(stack)):
        return "stack is empty"

    return stack.pop()

def solve(*args):
    """Implement your solution here."""
    pass


if __name__ == "__main__":
    # Example usage / test cases
    pass
