# lambda is equivalent to defining a function/method
# simple lambda (no parameter)
pp = lambda: print('Hello World')
# note: call lambda function by attaching parenthesis - ()
pp()  # this is redundant because you can as well directly print 'Hello World'

# displaying items of an iterable with a lambda expression
# note: no need for an outer print function as the items are
# already being printed by the lambda expression
print("Displaying items of a list below:")
list(map(lambda x: print(x), ['osagie', 'daisy', 'edoghogho']))
print("Displaying items of a list of list below:")
list(map(lambda x: print(x), [['osagie', 'daisy', 'edoghogho'], ['male', 'female', 'female']]))

# note: when using an if in a lambda exp, ALWAYS INCLUDE the else case
# otherwise it will throw a syntax error
print(f"Example 1: {(lambda x: 'Even Number' if not x%2 else 'Odd Number')(10)}")
print(f"Example 2: {(lambda x: 'Even Number' if x%2 == 0 else 'Odd Number')(3)}")
print(list(map(lambda x: 'Even Number' if x%2 == 0 else 'Odd Number', [10, 4, 13])))

# map lambda to iterable with single parameter
print('\nIterable with single parameter')
print(f'Example 1: {list(map(lambda x: x**2, [2, 4, 6]))}')
print(f"Example 2: {list(map(lambda x: 'A' in x, ['Anna', 'Beauty']))}")

# map lambda to iterable (with multiple parameters of unequal lengths)
# note: the result stops at the last item of smallest iterable (which is the last matching item in both)
print('\nIterable with multiple parameters')
print('Iterable parameters with unequal lengths')
# smallest list contains 2 items
print(f"Example 2: {list(map(lambda l, w: l.lower() in w.lower(), ['a', 'x', 'b', 'x'], ['Anna', 'Beauty']))}")
# smallest list contains 3 items
print(f"Example 2: {list(map(lambda l, w: l.lower() in w.lower(), ['a', 'x', 'c'], ['Anna', 'Beauty', 'Xavier', 'Christopher']))}")

# map lambda to iterable (with multiple parameters of equal lengths)
# note: secret is to pass a matching number into the iterable parameter
print('\nIterable with single parameter')
print('Iterable parameters with matching lengths')
print(f'Example 1: {list(map(lambda x, y: x**y, [2, 3, 5], [2, 2, 2]))}')
print(f"Example 2: {list(map(lambda l, w: l.lower() in w.lower(), ['a', 'b'], ['Anna', 'Beauty']))}")
