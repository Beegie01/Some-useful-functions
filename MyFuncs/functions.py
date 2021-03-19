def highest_val(iterable):
    '''
    to determine the largest number in a sequence
    '''

    largest_so_far = iterable[0]

    for num in iterable:
        if num > largest_so_far:
            largest_so_far = num

    print(largest_so_far)

    return largest_so_far


def smallest_val(iterable):
    '''
    to determine the smallest number in a sequence
    '''

    smallest_so_far = iterable[0]

    for num in iterable:
        if num < smallest_so_far:
            smallest_so_far = num

    print(smallest_so_far)

    return smallest_so_far


print('defining some variables')

c = 'Wale'
v = 'Osato'

names = ['Ade', 'Constance', 'Momoh', 'Osagie', 'Edoghogho']
gend = ['female', 'female', 'male', 'male', 'female']
ag =  [20, 26, 30, 28, 16]
num = [1.2, 33, 19]

students = {'student': names,
            'age': ag,
            'gender': gend}

genders = set(students['gender'])

mix = ('Ade', 20,'Constance', 26)



print('creating a caseless character-checker for variables')

def caseless_string_contains(var, char):
    "'checks for the number of times that an element appears in a variable, regardless of the case'"
    #STEPS INVOLVED:
	#step 1: create a placeholder to count the number of matches made
	#step 2: loop through given variable to access each element (string_character/ list_item)
	#step 3: when there's a match between a variable's element and the character we're checking for
	#step 4: increase the count value by 1 each time
	#step 5: At the end of the for loop,
	#step 6: confirm if the total count is 1
	#step 7: then print "hold up, char_name was found once"
	#step 8: or if the total count is more than 1
	#step 9: then print "hold up, char_name was found number of times"
	#step 10: otherwise (when total count is 0 or less),
	#step 11: then print "char_name not found"

    count = 0
    for elem in var:
        if elem.casefold() == char.casefold():
            count += 1

    #same indentation with for loop to indicate: at the end of the loop,
    if count == 1:
        return print(f"Hold up, {char} was found once!")
    elif count > 1:
        return print(f"Hold up, {char} was found {count} times!")
    else:
        return print(f'{char} was not found')


print("checking to see if an element appear within the variable")
source = input('variable')
check = input('check element: ')
caseless_string_contains(source, check)

#creating a case-sensitive character-checker for strings

#STEPS INVOLVED:
#step 1: create an empty container list (for our matched-up character)
#step 2: loop through string to get each item/element (ie individual characters)
#step 3: when there's a match between an element and the given character
#step 4: append the element to the empty list
#step 5: stop the current loop (because a match has been found and stored in our list)
#step 6: confirm if the list is not empty
#step 7: then print "hold up, it's in here"
#step 8: otherwise (when the list is still empty)
#step 9: then print "not in here"

def casesensitive_string_check(string, char):
    match = []
    for e in string:
        if e == char:
            match.append(e)
            break
    if len(match) > 0:
        return print("Hold up, it's in here")
    else:
        return print('Not in here')

casesensitive_string_check(source, check)


# check if a character is in a string

# STEPS INVOLVED:
# step 1: create an empty match list (for any matched-up character)
# step 2: loop through string to get each item/element (ie individual characters)
# step 3: when there's a match between an element and the given character
# step 4: append the element to the empty list
# step 5: break out of the current loop (because a match has been found and stored in our list)
# step 6: confirm if the list is not empty
# step 7: then print "hold up, it's in here"
# step 8: otherwise (when the list is still empty)
# step 9: then print "not in here"

def check_not(string, char):
    match = []
    for e in string:
        if e == char:
            match.append(e)
            break
    if len(match) > 0:
        return print("Sorry, there's a match")
    else:
        return print('Hooray!!!')

check_not(source, check)


def two_list_merger(list1, list2):
    "'takes each elements of two lists and merges them sequentially'"

    longer = max(len(list1), len(list2))
    container = []
    #tuple for lists of equal number of elements
    if len(list1) == len(list2):
        for num in range(len(list1)):
            container.append((list1[num], list2[num]))

    else:
        if len(list1) == longer:
            for ind,val in enumerate(list1):
                container.append(val)
                for ind2,val2 in enumerate(list2):
                    if ind == ind2:
                        container.append(val2)
        else:
            for ind,val in enumerate(list2):
                container.append(val)
                for ind2,val2 in enumerate(list1):
                    if ind == ind2:
                        container.append(val2)


    print(container)

    return container

print('merging two lists of equal length')
two_list_merger(['Ade', 'Bob', 'Nofe'],
	['female', 'male', 'female'])

print('merging two lists sequentially')
two_list_merger(['10', '20', 30], [33, '40', 'fall'])

print('merging two lists of unequal lengths')
two_list_merger(['God', 'too', 'jor!'], ['is', 'much'])


def get_mid_val(var):
    "'returns the element in the middle position'"

    #STEPS INVOLVED:
    #step 1: if the variable contains an even number of elements
    #step 2: mid-point of variable are those elements at: half of variable length minus one (due to Python's indexing starting from zero), up to one step ahead
    #step 3: otherwise (when the variable contains an odd number of elements)
    #step 4: mid-point of variable is the element at: half of variable length plus 0.5 (to make it a whole number), minus one (due to Python's indexing starting from zero)

    #when the variable has even number of length
    if len(var) % 2 == 0:
        halfway = int(len(var)/2) - 1 #note: by default Python returns answers as floats, thus we convert to int. Also Python starts counting from 0, so we minus one.
        print(var[halfway: (halfway+2)])
        return var[halfway:(halfway+2)]

    #when the variable has odd number of length
    else:
        halfway = int((len(var)/2 )+ 0.5) - 1
        print(var[halfway])
        return var[halfway]

nm = [1,2,8,3,5]
print('checking for middle value in this variable: {n}'.format(n=nm))
print(f"mid-point of {nm} is {get_mid_val(nm)}")

def add_ind(var):
    "'adds index to variable items: my attempt to replicate Python's enumerate() function'"
    indexing = range(len(var))
    container = []
    for num in indexing:
    	container.append((num, var[num]))
    print(container)
    return container


print('comparing add_index function to in-built enumerate function')

add_ind(['Abuja', 'Georgia', 'Alaska', 'Washington'])
list(enumerate(['Abuja', 'Georgia', 'Alaska', 'Washington']))


def sum13(nums):
    '''
    returns the sum of a list of numbers excluding
    13 and numbers occurring immediately after 13

    Example:

    a) sum13([1, 2, 2, 1, 13]) -->

    Output:
    7

    b) sum13([1, 2, 13, 4, 3, 13, 2, 2]) -->

    Output:
    8
    '''

    def count_13(nums):
        count = 0
        for num in nums:
            if num == 13:
                count += 1
        return count

    def occurs_last(nums):
        return nums[-1] == 13

    def len_less_than_3(nums):
        tot = sum(nums)
        c = count_13(nums)
        if len(nums) < 3:
            if c > 0 and occurs_last(nums):
                return sum(nums) - (c*13)
            elif c > 0 and occurs_last(nums) == False:
                return 0
            else:
                return tot

    tot = 0

    # for lists that have less than three items
    if len(nums) < 3:
        return len_less_than_3(nums)

    # for lists that have 3 or more items
    else:
        for n in range(len(nums)):

            if (n == 0) or (nums[n] == 13) or (nums[n-1] == 13):
                print(f"Ignoring {nums[n]} at {n}")
                continue
            tot += nums[n]
        if nums[0] != 13:
            tot += nums[0]

    return tot
