# DEFINED FUNCTIONS ARE:
# float_inp, num_inp, name_inp, password_inp,
# ask_next, ask_to_save, generate_password, save_progress
# retrieve, game_mode

import string
from datetime import datetime
from mynum_funcs import float_range
import random, re, sys, os

def play_on():

    while True:
        prompt = "\nPress Enter to continue:\n>\t"
        val = input(prompt)

        acc_range = ''
        if val.lower() not in acc_range:
            print(f"\n{val} is not valid!")
            continue

        break


def exit_play():

    while True:
        prompt = "\nTo exit game, enter 'e'\nType 'c' and press Enter to continue:\nCONTINUE/EXIT PLAY>\t"
        val = input(prompt)
        acc_range = ['c', 'continue', 'e', 'exit']

        if val.lower() not in acc_range:
            print(f"\n{val} is not valid!")
            continue

        if val.lower() in ['e', 'exit']:
            return True
        else:
            return False

def float_inp(prompt, lim=None, step=None):
    '''
    to collect numeric user input and output a float
    '''

    if lim != None or type(lim) == tuple:
        lower_lim,upper_lim = lim
        acc_range = float_range(lower_lim,upper_lim,step)

    else:
        acc_range = None


    while True:
        inp = input(prompt)
        try:
            num = float(inp)
        except ValueError as err1:
            print(f"\n\n{inp} is not a valid number!\nPlease enter an number")
            continue

        if acc_range != None:
            if num not in acc_range:
                print(f"\n{num} is out of range!")
                continue

        return num


def num_inp(prompt, lim=None):
    '''
    to collect numeric user input and output a float
    '''

    if lim != None:
        lower_lim,upper_lim = lim
        acc_range = range(lower_lim,upper_lim)

    else:
        acc_range = None


    while True:
        inp = input(prompt)
        try:
            num = int(inp)
        except ValueError as err1:
            print(f"\n\n{inp} is not an integer!\nPlease enter a whole number")
            continue

        if acc_range != None:
            if num not in acc_range:
                print(f"\n{num} is out of range!")
                continue

        return num

def name_inp(name_prompt):
    '''
    to collect alpha-numeric user input
    '''

    while True:

        inp = input(name_prompt)

        if inp in string.punctuation:
            print("Error: Invalid Entry!")
            continue

        print(f"\n\n{inp.title()} Entered!")
        return inp.title()

def password_inp(pwrd_prompt):
    '''
    to collect password from user
    input has no spec
    '''
    inp = input(pwrd_prompt)

    print(f"\n\n{inp.title()} Entered!")
    return inp


def ask_next():
    print("\n\nTo continue, enter 'y'\nTo stop, enter 'n'")
    prompt = '\nHave another?\nYes/No>\t'
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

    acc_range = ['save', 's', 'no', 'n']

    prompt = "To save game progress, type 's' & press Enter\nTo exit without saving, type 'n' and press Enter:\nSAVE/EXIT>\t"

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
    output is a 7 character long password
    '''

    SEP1,SEP2 = "*12345", "$67890"
    LEFT,RIGHT = 'aAbBcCdDeEfF', 'zZlLhHpPmMkK'
    return f"{random.choice(SEP1)}{random.choice(LEFT)}{random.choice(LEFT)}{random.choice(range(100, 1000))}{random.choice(RIGHT)}{random.choice(RIGHT)}{random.choice(SEP2)}"

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


def retrieve(self):
    '''
    to retrieve saved app/game records from file
    '''

    # ask for the saved usernames
    prompt = "\nEnter password:\n>\t"
    pword = password_inp(prompt)


    # retrieve stored information from file
    hand = open('C:\\Users\\welcome\\Desktop\\SimplePythonChallenges\\RockPaperScissors\\game_data.py')

    # file data contains a list of dictionary pairs
    file_data = hand.read().strip()

    # declaring useful array variables
    passwords = {}
    draws = []
    usernames = []
    wins = []
    losses = []


    # for each dictionary list in the file
    for n, dic in enumerate(file_data.split("\n")):
        # eliminating the first line containing the comment on column order
        if n == 0:
            continue

        # transform the dict saved in str format back into dict type
        di = eval(dic)

        # here, each key is a unique password
        # value is a list of list containing each player's records
        for k,v in di.items():

            # collecting and indexing the passwords from the file
            if k == pword:
                print("Password Found!")

                # here value is the list of lists containing each player's records
                for each_player_rec in v:
                    # segmenting the information onto separate lists
                    usernames.append(each_player_rec[0]), wins.append(each_player_rec[1]), losses.append(each_player_rec[2]), draws.append(each_player_rec[3])

    # when username was collected
    if len(usernames) < 1:
        print(f"Password: {pword} is not on record\nPlease check for incorrect spelling")
        return "Not Found"

    else:
        # after match has been found
        print(f"Users: {usernames}\nWins: {wins}\nLosses: {losses}")
        self.p1['username'], self.p2['username'] = usernames[0], usernames[1]
        self.p1['games_won'], self.p2['games_won'] = wins[0], wins[1]
        self.p1['games_lost'], self.p2['games_lost'] = losses[0], losses[1]
        self.p1['draws'], self.p2['draws'] = draws[0], draws[1]

        print("\n\nPlayers' records have been retrieved and restored!")
        return 'Done'

def game_mode():

    acc_range = ['new', 'n', 'c', 'cont']
    while True:
        # ask to start new or continue saved game
        prompt = "\n\nTo start new game, enter 'new'\nTo continue saved game, enter 'cont'\nNEW\CONTINUE>\t"
        inp = input(prompt)

        if inp[0].lower() not in acc_range:
            print("\nPlease select 'new' or 'cont'!")
            continue

        return inp[0].lower()
