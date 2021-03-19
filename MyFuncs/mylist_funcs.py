from collections import defaultdict, OrderedDict


def sorter(num_list):
    '''
    returns a sorted a list of given numbers
    '''

    sorter = []

    while len(num_list) != 0:
        # assume smallest number is the first element on list
        min_num = num_list[0]
        dd = {}

        for ind, num in enumerate(num_list):

            if min_num < num:
                continue
            # when current num is smaller than or equal to the previous minimum number
            # the index of the current number is recorded
            dd['index'] = ind

        # after the smallest number and its index has been recorded
        # delete the minimum number from list
        # and look for the new minimum number
        sorter.append(num_list.pop(dd['index']))
    return sorter


def index_iters(dd):
    '''
    to convert a list or list of lists into a dictionary
    with each list having the corresponding row number as its key


    Example:

    a = ['cat', 'lab', 'dog', 'sand', 'goat']
    b = (1,3,5,6)
    sd = {'a': 'hello', 'b': 'ell', 'c': 'today'}
    c = [a,b,sd]

    index_row(c) --> indexed dict

    Output:
    {1: ['cat', 'lab', 'dog', 'sand', 'goat'], 2: (1, 3, 5, 6), 3: {'a': 'hello', 'b': 'ell', 'c': 'today'}}
    '''
    d = {}
    count = 0
    check = None

    # ascertain the data structure of the input
    # if it is not a list of lists --> True
    for x in dd:
        if type(x) not in [list, tuple]:
            check = True

    # for a list or tuple
    if check:
        count += 1
        d[count] = dd

    # for list of lists
    else:
        for lst in dd:
            count += 1
            d[count] = lst
    return d


def iter_join(*args):
    '''
    combines the values of lists, tuples, dict and sets
    into one single list

    Example:
    a = ['cat', 'lab', 'dog', 'sand', 'goat']
    e = (1,3,5,7,9)

    join_lists(e,a) -->

    Output:
    [1, 3, 5, 7, 9, 'cat', 'lab', 'dog', 'sand', 'goat']
    '''

    lst = []

    if type(args[0]) not in [list, tuple, dict, set]:
        return print('Error: List or Tuple was not give')

    for i in args:
        lst.extend(i)

    return lst


def minmax_range(nums):
    '''
    returns the minimum and maximum values of a list of numbers
    as well as the range between them

    Example:
    ages= [16, 27, 34, 77, 32, 23, 26, 21, 40, 55, 62]
    minmax_range(ages) -->

    Output:
    'Max: 77, Min: 16, Range: 61'
    '''

    mx = nums[0]
    mn = nums[0]

    for n in nums:
        if n > mx:
            mx = n
        if n < mn:
            mn = n
    return f"Max: {mx}, Min: {mn}, Range: {mx - mn}"


def word_lengths(L):
    '''
    returns lengths of words in a given list

    Example:

    names = ['Edoghogho', 'Osagie', 'Elliot', 'Ekpen', 'Ugiomon', \
    'Omosede', 'Eugenia', 'Emma', 'IK', 'Dan', 'Justina']

    word_lengths(names)

    Output:
    [9, 6, 6, 5, 7, 7, 7, 4, 2, 3, 7]
    '''
    return list(map(len, L))


def w_length(phrase):

    word_list = phrase.split()
    counter = []

    for w in word_list:
        counter.append(len(w))
    return counter


def filter_words(word_list, letter):
    '''
    to filter a list of words
    for presence of a given letter
    returns a list of only words containing the given letter

    Example:
    l = ['hello','are','cat','dog','ham','hi','go','to','heart', 'cheap', 'sheep']

    filter_words(l, 'o') -->

    Output:
    ['hello', 'dog', 'go', 'to']
    '''
    # for each word in the list of words
    for word in word_list:
        # lambda expression returns True, if a word contains the given letter
        # filter the list of words based on the output of the lambda expression
        return list(filter(lambda word: letter in word, word_list))


def containing_letter(word_list, letter):
    '''
    similar in result to the filter_words function
    '''
    lst = []
    for word in word_list:
        if letter in word:
            lst.append(word)
    return lst


def ind_list(L):
    '''
    returns a dict of list elements as keys
    and index as values

    Example:
    d_list(['a','b','c'])

    Output:
    {'a': 0, 'b': 1, 'c': 2}
    '''
    dd = {}
    # index(n) and element (l)
    for n,l in enumerate(L, start=0):
        # assign l as dict-key and n as dict-value
        dd[l] = n

    return dd


def list_ind(L):
    '''
    using dict comprehension
    '''
    return {ele:ind for (ind, ele) in enumerate(L)}


def count_ele(*args):
    '''
    return the count of elements in a container

    Example 1:
    gender = ['Female', 'Male', 'Male', 'Male', 'Female', \
    'Female', 'Female', 'Male', 'Male', 'Male', 'Female']


    count_ele(gender) -->

    Output:
    {'Female': 5, 'Male': 6}

    Example 2:
    lst1 = [1,1,1,1,4,4,4,5,5,6,6,6,6,6,2,2]

    count_ele(lst1, gender) -->

    Output:
    {1: 4, 4: 3, 5: 2, 6: 5, 2: 2, 'Female': 5, 'Male': 6}
    '''


    dd = {}

    try:
        for l in args:
            for ele in l:
                if ele not in dd:
                    dd[ele] = 1
                else:
                    dd[ele] += 1
    except TypeError:
        return len(lst)

    return dd


def counter_od(*args):
    '''
    returns the count of elements as an ordered dict
    where keys are list values and values are the count

    Example:
    ss = deque(['Osagie', 'Beegie', 'Dazzler', 'Johnny', 'Aiby', 'Great', 'Imose', 'Amah', 'Great'])

    counter(ss) -->

    Output:
    OrderedDict([('Great', 2),
             ('Osagie', 1),
             ('Johnny', 1),
             ('Imose', 1),
             ('Dazzler', 1),
             ('Beegie', 1),
             ('Amah', 1),
             ('Aiby', 1)])
    '''

    dfd = defaultdict(lambda: 0)

    for lst in args:

        for ind in range(len(lst)):
            dfd[lst[ind]] += 1

    li = []
    di = OrderedDict()
    for (k,v) in dfd.items():
        li.append((v,k))

    li.sort(reverse=True)
    for (v,k) in li:
        di[k] = v

    return di


def counter(*args):
    '''
    similar in function to the counter func

    Example:
    ss = deque(['Osagie', 'Beegie', 'Dazzler', 'Johnny', 'Aiby', 'Great', 'Imose', 'Amah', 'Great'])

    gender = ['Female', 'Male', 'Male', 'Male', 'Female', \
    'Female', 'Female', 'Male', 'Male', 'Male', 'Female']

    counter(ss, gender) -->

    Output:
    [('Male', 6),
     ('Female', 5),
     ('Great', 2),
     ('Osagie', 1),
     ('Beegie', 1),
     ('Dazzler', 1),
     ('Johnny', 1),
     ('Aiby', 1),
     ('Imose', 1),
     ('Amah', 1)]
    '''
    from collections import defaultdict

    dfd = defaultdict(lambda: 0)

    for lst in args:

        for ind in range(len(lst)):
            dfd[lst[ind]] += 1

    di = sorted(dfd.items(), key=lambda x: x[1], reverse=True)

    return di
