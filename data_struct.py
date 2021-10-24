import collections as c, array

print(int(), int()+1)

# beautiful way to delete a variables, objects, items, etc
# m = [12, 'one', 'apple']
# print(m)
# del(m[0])  # or del m[0]
# print(m)

# another way to define a dict
# aa = dict(gender='male', name='Osagie')
# print(aa)

# namedtuple is equivalent to a class that only has attributes -  no methods
# student = c.namedtuple('Student', ['name', 'gender', 'age', 'course'])

# mike = student(name='Mike', gender='M', age=21, course='Medicine')

# print(mike.name)

# dd = {'name': 'Mike', 'gender': 'M', 'age': 21}

# print(dd)

# ordered dict may appear as a list of tuple, BUT you can only access its elements through its keys
# most imp, it is order sensitive (retains its order of insertion)
# od = c.OrderedDict(dd)
# del dd['gender']  # or del(dd['gender'])
# dd.popitem()

# print(od)
# od.move_to_end('gender')

# od['height'] = '200m'  # each new addition is inserted at the end and retains its position

# del(od['age'])  # beautiful way to delete a variable

# print(od)

# dd2 = {'name': 'Ann', 'gender': 'F', 'age': 23}

# print(f'Ordered dict: {od.__sizeof__()}\nDict (normal): {dd.__sizeof__()}')  # ordered dict is larger than normal dict


# cm = c.ChainMap(dd, dd2)
# print(cm, cm.items())
# for k, v in cm.items():
#     if k == 'name' and v == 'Mike': print(v)
#
# print((lambda x, m: str(m) if x == 'name' and m == 'Ann' else None)(cm.keys(), cm['name']))
#

# counter is user to count number of repetitions of items of an iterable
# presents result in a dict
# nums = [2, 33, 2, 45, 5, 5, 33, 5]
# alps = ['a', 'b', 'c', 'b', 'd', 'b', 'd', 'c']
# print(c.Counter('hello how are you doing dear?'))
# co = c.Counter(alps)
# print(co)

# get top two highest items
# print(f'Most common: {co.most_common(2)}')

# create a list from counter's items
# print(list(co.elements()))

# another way to create a counter (kwargs style)
# co2 = c.Counter(b=1, c=2, d=2)
# print(co2)

# operations between counters
# diff = co - co2
# print(diff)

# number of items
# print(co.values(), co2.values())
# print(sum(co.values()), sum(co2.values()))

# deque
dq = c.deque('bcd'); print(dq)
dq.append('e'); print(dq)  # append to end of list (right side - default)
dq.appendleft('a'); print(dq)  # append to the beginning of list (left side)

dq.extend('fgh'); print(dq)
dq.extend('210'); print(dq)
# print(dq.index('2'))
# print(dq[8])
s = dq.index('2')
# print(type(s))
# print(dq[1:5])  # Error: deque cannot be sliced
for e in '210':  # delete 210 for deque
    del(dq[dq.index(e)])
    print(dq)

dq.extendleft('210'); print(dq)  # note: inserts 2 first to left, then 1, and lastly 0
dq.popleft(); print(dq)

# rotate
dq.rotate(); print(dq)
dq.rotate(-1); print(dq)


# default dict
my_list = ['Alice', 'Bob', 'Chris', 'Xavier', 'Bill', 'Ashley', 'Anna']
# group items according to their first element
dd = c.defaultdict(list); print(dd)  # data type not value
for name in my_list:
    dd[name[0]].append(name)
print(dd.items()); print(dd['A'], dd['B'])
# dd = {}
# for name in my_list:
#     if name[0] not in dd:
#         dd[name[0]] = []
#     dd[name[0]].append(name)
# print(dd.items())

# for counting
dd = c.defaultdict(int); print(dd)
for name in my_list:
    dd[name[0]] += 1
print(dd.items())