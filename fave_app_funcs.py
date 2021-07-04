# DEFINED FUNCTIONS ARE:
# float_inp, num_inp, name_inp, password_inp,
# ask_next, ask_to_save, generate_password, save_progress
# retrieve, game_mode

import string, random, csv, getpass, os
# from datetime import datetime

from mynum_funcs import float_range


def play_on():
    '''
    does nothing when Enter is pressed
    and continues to the next sequence of events
    '''

    while True:
        prompt = "\nPress Enter to continue:\n>\t"
        val = input(prompt)

        acc_range = ''
        if val.lower() not in acc_range:
            print(f"\n{val} is not valid!")
            continue

        break


def exit_play():
    '''
    returns True if exit/x is Entered
    and False if continue/c is Entered
    '''

    while True:
        prompt = "\nTo exit game, enter 'x'\nType 'c' and press Enter to continue:\nCONTINUE/EXIT PLAY>\t"
        val = input(prompt)
        acc_range = ['c', 'continue', 'x', 'exit']

        if val.lower() not in acc_range:
            print(f"\n{val} is not valid!")
            continue

        if val.lower() in ['x', 'exit']:
            return True
        else:
            return False


def sentence_inp(prompt):
    '''
    returns a str of sentences
    and capitalises the first words in each sentence (if plural)
    '''
    while True:
        inp = input(prompt)

        if not(inp.isascii()):
            print("Error: Invalid Entry!")
            continue

        print(f"\n\nEntered:\n\t{'. '.join([l.strip().capitalize() for l in inp.split('.')])}")
        # capitalise first words in the sentence
        return '.\n'.join([l.strip().capitalize() for l in inp.split('.')])


def float_inp(prompt, lim=None, step=None):
    '''
    to collect numeric user input
    output is a floating point number
    Optional args:
    lim: tuple of upper and lower limit can be given
    to specify a value within the upper and lower values
    step: specifies a step size (in float) of range for the given lim
    '''

    if lim is not None or type(lim) == tuple:
        lower_lim, upper_lim = lim
        acc_range = float_range(lower_lim, upper_lim, step)

    else:
        acc_range = None

    while True:
        inp = input(prompt)
        try:
            num = float(inp)
        except ValueError:
            print(f"\n\n{inp} is not a valid number!\nPlease enter an number")
            continue

        if acc_range is not None:
            if num not in acc_range:
                print(f"\n{num} is out of range!")
                continue

        return num


def num_inp(prompt, lim=None):
    '''
    to collect numeric user input
    output is an integer
    Optional args:
    lim: tuple of upper and lower limit (int) can be given
    to specify a value within the upper and lower values
    '''

    if lim is not None:
        lower_lim, upper_lim = lim
        acc_range = range(lower_lim, upper_lim)

    else:
        acc_range = None

    while True:
        inp = input(prompt)
        try:
            num = int(inp)
        except ValueError:
            print(f"\n\n{inp} is not an integer!\nPlease enter a whole number")
            continue

        if acc_range is not None:
            if num not in acc_range:
                print(f"\n{num} is out of range!")
                continue
        return num


def name_inp(name_prompt):
    '''
    to collect alpha-numeric user input
    output is a title-cased str
    '''

    while True:

        inp = input(name_prompt)

        if inp in string.punctuation:
            print("Error: Invalid Entry!")
            continue

        print(f"\n\n{inp.title()} Entered!")
        return inp.title()


def password_inp(prompt):
    '''
    to collect password from user
    input has no spec and not visible onscreen
    '''

    inp = getpass.getpass(prompt)

    print(f"\n\n{inp.title()} Entered!")
    return inp


def file_path_inp():
    '''
    '''
    prompt1, prompt2, prompt3 = "\n\nEnter folder path\nNOTE: Use double foward slashes to separate parent from child folder\nABSOLUTE FOLDER PATH>\t", "\n\nEnter file name (without extension)\nFILE NAME>\t", "\n\nEnter File extension (e.g csv, txt)\nFILE EXTN>\t"

    while True:
        folder_path = input(prompt1)

        try:
            for child in os.listdir(folder_path):
                print(child)
            break
        except FileNotFoundError as FNFE:
            print(f"Error: Invalid Entry!\n{FNFE}")
            continue

    file_name, extn = input(prompt2), input(prompt3)

    print(f"{folder_path}\\{file_name}.{extn} Entered!")
    return folder_path, file_name, extn

def ask_next(pmt):
    '''
    return True if yes/y is Entered
    and False if no/n is Entered
    input is case insensitive
    '''
    print(f"\n\n{pmt}")
    prompt = '\nYes/No>\t'
    acc_range = ['no', 'n',  'yes', 'y']

    while True:
        inp = input(prompt)

        if inp.lower() not in acc_range:
            print("Error: Entry is invalid!")
            continue

        if inp.lower() in ['yes', 'y']:
            return True
        else:
            return False


def ask_to_save():
    '''
    returns True if save/s is Entered
    and False if ex/x is Entered
    '''

    acc_range = ['save', 's', 'ex', 'x']

    prompt = "To save game progress, type 's' & press Enter\nTo exit without saving, \
    type 'x' and press Enter:\nSAVE/EXIT>\t"

    while True:
        inp = input(prompt)

        if inp.lower() not in acc_range:
            print("Error: invalid entry!")
            continue

        if inp.lower() in ['save', 's']:
            return True
        return False


def generate_password():
    '''
    simply generates a random password
    returns an output that is 9 characters long
    consisting of digits, alphabets, and symbols

    EXAMPLE:
    generate_password() -->

    OUTPUT:
    '0AB889zZ$'
    '''

    sep1, sep2 = "0123456789", "*!@#$%&~"
    left, right = 'aAbBcCdDeEfFHhGgJjKk', 'zZLlQqTtYypPmMkKRrSs'
    return f"{random.choice(sep1)}{random.choice(left)}{random.choice(left)}\
{random.choice(range(100, 1000))}{random.choice(right)}{random.choice(right)}{random.choice(sep2)}"


def save_progress(file_path, obj):
    '''
    to save game/app records
    input are file path of text file for saving
    and class instance/object
    '''

    password = generate_password()

    hand = open(file_path, 'a')

    # stored fields include
    # {password: [[username1, games_won, games_lost, games_drawn], [username2, games_won, games_lost, games_drawn]]}
    info = f"\n{dict( [ ('password', password), ('obj_data', obj.__dict__ ) ])}"

    hand.write(info)

    hand.close()

    print(f"\n\nGame saved!\nTo continue use the password below:\n\nPASSWORD>\t{password}")


def save_details(file_path, obj):
    '''
    to save game/app records
    input are file path of text file for saving
    and class instance/object
    '''
    # generate a 9-digit password
    password = generate_password()

    # storing object attributes as a variable
    info = (password, obj.__dict__)

    with open(file_path, 'a', newline='') as h:

        # csv writer object
        csv_wr = csv.writer(h)

        # insert new row into csv writer object
        csv_wr.writerow(info)

    print(f"\n\nGame saved!\nTo continue use the password below:\n\nPASSWORD>\t{password}")


def retrieve(data_file):
    '''
    to retrieve saved app/game records from the given file
    or filepath
    '''

    # ask for the saved usernames
    prompt = "\nEnter password:\n>\t"
    p_word = password_inp(prompt)

    # retrieve stored information from file

    # each line in file data contains a pair of password and its corresponding dictionary of attributes
    with open(data_file) as h:

        # read the file content with csv reader function
        csv_read = csv.reader(h)

        # creating a list of values read from file
        data_list = list(csv_read)

    # checking each row of data file
    for line in data_list:

        # if given password is matched
        if line[0].lower() == p_word.lower():
            # print(line[1])
            attr = eval(line[1])
            print("\n\nPlayers' records have been found")
            return attr

    print(f"\n\nPassword: {p_word} is not on record\nPlease check for incorrect spelling")
    return False


def game_mode():

    acc_range = ['new', 'n', 'c', 'cont']
    while True:
        # ask to start new or continue saved game
        prompt = "\n\nTo start afresh, enter 'new'\nTo continue, enter 'cont'\nNEW or CONTINUE>\t"
        inp = input(prompt)

        if inp[0].lower() not in acc_range:
            print("\nPlease select 'new' or 'cont'!")
            continue

        return inp[0].lower()

def uniqueid():
    '''
    randomly generates a ten-digits integer
    :return: a generator that increments by 1
    '''
    start = rm.getrandbits(32)  # return a random number of ten digits
    while True:
        yield start  # return the random ten-digits
        start += 1  # increment the random ten-digit number by 1

def convert_dictlist_to_tuplelist(dict_list: dict):
    '''
    to convert a list of dictionaries into a list of tuples
    EXAMPLE:
    convert_dictlist_to_tuplelist(["{'age': 12, 'gender': 'male'}", "{'age': 10, 'gender': 'female'}"])

    output -->
    [[('age', 12), ('gender', 'male')],
    [('age', 10), ('gender', 'female')]]
    '''
    if type(dict_list) not in [list, tuple]:
        print('Parameter must be a list or tuple')
        return None
    if [True for dd in dict_list if type(dd) != dict and type(eval(dd)) != dict]:
        print('List must contain dictionaries')
    rec = []
    for ind in range(len(dict_list)):
         rec.append([(k, v) for k, v in eval(dict_list[ind]).items()])
    return rec

def tabulate_array(header: list or tuple=[], *cols: list or tuple):
    '''
    create a table with given lists/tuple as each column/field
    and length of each list/tuple as number of rows
    :param args: list or tuple
    :return:
    '''
    col_len = len(cols)
    row_lens = [len(var) for var in cols]
    print(f'Columns: {col_len}\nRows: {row_lens}')

    import tkinter as tk
    from tkinter import scrolledtext

    root = tk.Tk()
    f = tk.Frame(root, bg='orange')
    f.grid()

    if not header:
        for i in range(col_len):  # for number of columns
            for l in cols:  # for every list/tuple given
                for j in range(len(l)):  # for every row in each list/tuple
                    e = tk.Entry(f)
                    e.grid(row=j, column=i)
                    e.insert(0, cols[i][j]); e['state'] = 'disabled'
    else:
        for i in range(len(header)):  # for number of columns
            e = tk.Entry(f)
            e.grid(row=0, column=i)
            e.insert(0, header[i].upper());
            e['state'] = 'disabled'
        for i in range(col_len):
            for l in cols:  # for every list/tuple given
                for j in range(len(l)):  # for every row in each list/tuple
                    e = tk.Entry(f)
                    e.grid(row=j+1, column=i)  # reserve first row for column headers
                    e.insert(0, cols[i][j]); e['state'] = 'disabled'

    root.mainloop()


# names, ages, genders = ['Osagie', 'Jerry', 'Cyndy'], [25, 12, 34], ['M', 'M', 'F']
# header = ['names', 'ages', 'genders']
# data = [names, ages, genders]
#
# tabulate_array(header, *data)