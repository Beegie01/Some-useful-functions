import string

text = "Is "#this or that, the major (or minor) thing to check? I'd say yes"

def clean_split(text):
    '''
    returns a list of words in a string

    Example:
    q = "Is this love that I'm feeling, I wan't to know now?"

    split(q) -->

    Output:
    ['Is', ' this', ' love', ' that', " I'm", ' feeling', ' I', " wan't", ' to', ' know']
    '''
    clean_text = ''
    # to erase punctuation marks
    for char in text:
        if char in "?.,)()":
            continue
        clean_text += char

    # to cut out words
    whitespaces = ' \n\t'
    coordinates = []
    word_list = []

    # capturing the indices of each whitespaces in the text
    for ind in range(len(clean_text)):
        if clean_text[ind] in whitespaces:
            coordinates.append(ind)

    # in case there is no whitespace at end of text
    if coordinates[-1] != len(clean_text):
        coordinates.append(len(clean_text))

    # to map out each words according to list of coordinates/coordinates
    for num, ind in enumerate(coordinates):
        if num == 0:
            word_list.append(clean_text[:coordinates[num]])
            continue
        word_list.append(clean_text[(coordinates[num-1] + 1 ) : coordinates[num]])

    return word_list


text = "Is "#this or that, the major (or minor) thing to check? I'd say yes"

def split(text, sep):
    '''
    returns a list of words in a string
    split at given seperator locations

    Example:
    q = "Please buy me 2 packs, 3 drums, 5 shells, 1 satchet"

    split(q, ',') -->

    Output:
    ['Please buy me 2 packs', ' 3 drums', ' 5 shells', ' 1 satchet']
    '''

    if sep not in text:
        print()
        return f"'{sep}' not found in text"

    coordinates = []
    word_list = []

    # capturing the indices of each whitespaces in the text
    for ind in range(len(text)):
        if text[ind] in sep:
            coordinates.append(ind)

    # in case there is no whitespace at end of text
    if coordinates[-1] != len(text):
        coordinates.append(len(text))

    # to map out each words according to list of coordinates/coordinates
    for num, ind in enumerate(coordinates):
        if num == 0:
            word_list.append(text[:coordinates[num]])
            continue
        word_list.append(text[(coordinates[(num-1)]+1) : coordinates[num]])

    return word_list


def words_count(phrase):
    '''
    counts the number of occurrences in a string of words

    Example:
    words_count("How low can you go in this club? Baby, I know you can go very low ... winks.") -->

    Output:
    [(2, 'you'), (2, 'low'), (2, 'go'), (2, 'can'), (1, 'winks.'), \
    (1, 'very'), (1, 'this'), (1, 'know'), (1, 'in'), (1, 'club?'), \
    (1, 'I'), (1, 'How'), (1, 'Baby,'), (1, '...')]
    '''

    new_phrase = ''
    # sieve of the punctuation marks
    for char in phrase:
        if char in string.punctuation:
            continue
        new_phrase += char

    # split up new phrase into a list of words
    word_list = new_phrase.split()
    counter = {}

    # count each word (in lower case)
    for w in word_list:
        if w.lower() not in counter:
            counter[w] = 1
            continue
        counter[w] += 1

    sec = []
    fin = []

    for k,v in counter.items():
        sec.append((v,k))

    sec.sort(reverse=True)    # alternatively: sorted_list = sorted(sec, reverse=True)

    for v,k in sec:
        fin.append((k,v))
    return fin



def gen_ALP():
    '''
    generates ascii_uppercase letters
    '''
    for n in range(len(string.ascii_letters)):
        if string.ascii_letters[n] not in string.ascii_uppercase:
            continue
        yield string.ascii_letters[n]

def gen_alp():
    '''
    generates ascii_lowercase letters
    '''
    for n in range(len(string.ascii_letters)):
        if string.ascii_letters[n] not in string.ascii_uppercase:
            continue
        yield string.ascii_letters[n]
