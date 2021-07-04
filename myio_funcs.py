import os

def file_search(folder_name, search_pattern: str=None, file_ext: str=None):
    '''
    returns the full path/location of the file that
    matches the given search pattern and extension
    '''
    if search_pattern is None and file_ext is None:
        raise ValueError('No valid argument given for search_pattern and file_ext parameter')
    
    if folder_name is None:
        raise ValueError('Folder_name to serve as starting point for search is not given')
        
    result = {}
    for dirpath, folders, files in os.walk(folder_name):

        for file in files:
            if file_ext is not None and search_pattern is not None:  # when both file_ext and search_pattern are both known
                if search_pattern.lower() in file.lower() and file_ext.lower() in file.lower():
                    result.setdefault(file, None)
                    result[file] = f'{dirpath}\\{file}'
            elif file_ext is not None and search_pattern is None:  # when only file_ext is known
                if file_ext.lower() in file.lower():
                    result.setdefault(file, None)
                    result[file] = f'{dirpath}\\{file}'
            else:  # when only search_pattern is known
                 if search_pattern.lower() in file.lower():
                    result.setdefault(file, None)
                    result[file] = f'{dirpath}\\{file}'

    return result


def ds_modules_importer():
    '''
    import the following data science modules:
    pandas as pd, numpy as np, seaborn as sns
    matplotlib.pyplot as plt
    :return: pd, np, plt, sns
    '''
    import pandas as pd, numpy as np
    import matplotlib.pyplot as plt, seaborn as sns
    print('\n\nData Science Modules imported!\n'
          'numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns\n')
    return pd, np, plt, sns


def read_allfile_content(folder_path=None, file_name=None, extn=None):
    '''
    copies out the content of a particular file (if file name is specified)
    or all files within specified folder
    Inputs:
    -path of folder containing file,
    -name of file(without extension),
    -file extension
    Returns: a list of strings/lines
    each list element corresponds to a line in the file
    '''
    lines = []
    extns = ['.txt', '.csv']

    # when no argument was passed
    if (folder_path is None) and (file_name is None) and (extn is None):

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

        ws = ['', ' ']

        if folder_path in ws:
            folder_path = None

        if file_name in ws:
            file_name = None

        if extn in ws:
            extn = None

    # print(f"Folder: {folder_path}\nFile: {file_name}\nExtension: {extn} Entered!")

    # when at least one argument was passed
    # when parent folder path was specified
    if folder_path is None:
        for path, folders, files in os.walk(os.getcwd()):
            # and both the file name and extension were also specified
            if len([True for ele in [file_name, extn] if ele is not None]) == 2:
                 for file in files:
                     if file.lower() == file_name.lower()+'.'+extn.lower():
                         print(f"\nFound {file} at\n{path}\n")
                         try:
                             with open(path+"\\"+file, 'r+', encoding='utf8') as hand:
                                 text = hand.readlines()
                             print(text)
                             lines.extend(text)
                             break
                         # when file cannot be read
                         except UnicodeDecodeError as UDE:
                            print(f"Error when reading {file}\n{UDE}")
                            continue

            # when parent folder path was specified and only the file name was specified
            elif (file_name is not None) and (extn is None):
                for file in files:
                    if file_name.lower() in file.lower():
                        print(f"\nFound {file} at\n{path}\n")
                        try:
                            with open(path+"\\"+file, 'r+', encoding='utf8') as hand:
                                text = hand.readlines()
                            print(text)
                            lines.extend(text)
                        # when file cannot be read
                        except UnicodeDecodeError as UDE:
                            print(f"Error when reading {file}\n{UDE}")
                            continue

            # when parent folder path was specified and only extension is specified
            elif (extn is not None) and (file_name is None):
                for file in files:
                    if '.'+extn.lower() in file.lower():
                        print(f"\nFound {file} at\n{path}\n")
                        try:
                            with open(path + "\\" + file, 'r+', encoding='utf8') as hand:
                                text = hand.readlines()
                            print(text)
                            lines.extend(text)
                        # when file cannot be read
                        except UnicodeDecodeError as UDE:
                            print(f"Error when reading {file}\n{UDE}")
                            continue

    # when parent folder path was specified
    else:
        for parent_folder_path, sub_folders, files in os.walk(folder_path):

            # to copy all text files in the specified directory
            # but both the file name and extension were not specified
            if (file_name is None) and (extn is None):
                for file in files:
                    # search for files with .txt and .csv extensions
                    if [True for ext in extns if ext in file.lower()]:
                        print(f"\nFound {file} at\n{parent_folder_path}\n")
                        # open file in read and write mode
                        try:
                            with open(parent_folder_path + "\\" + file, 'r+', encoding='utf8') as hand:
                                text = hand.readlines()
                            print(text)
                            lines.extend(text)
                        # when file cannot be read
                        except UnicodeDecodeError as UDE:
                            print(f"Error when reading {file}\n{UDE}")
                            continue

            # when only the file name was specified
            elif (file_name is not None) and (extn is None):
                for file in files:
                    if file_name.lower() in file.lower():
                        print(f"\nFound {file} at\n{parent_folder_path}\n")
                        try:
                            with open(parent_folder_path + "\\" + file, 'r+', encoding='utf8') as hand:
                                text = hand.readlines()
                            print(text)
                            lines.extend(text)
                        # when file cannot be read
                        except UnicodeDecodeError as UDE:
                            print(f"Error when reading {file}\n{UDE}")
                            continue

            # but only the file extension was specified
            elif (extn is not None) and (file_name is None):
                for file in files:
                    # only search for files with the specified extension
                    if '.'+extn in file.lower():
                        print(f"\nFound {file} at\n{parent_folder_path}\n")
                        # open file in read and write mode
                        try:
                            with open(parent_folder_path + "\\" + file, 'r+', encoding='utf8') as hand:
                                text = hand.readlines()
                            print(text)
                            lines.extend(text)
                        # when file cannot be read
                        except UnicodeDecodeError as UDE:
                            print(f"Error when reading {file}\n{UDE}")
                            continue

            # when file name and extension were also specified
            elif (file_name is not None) and (extn is not None):
                for file in files:
                    if file.lower() == file_name.lower()+'.'+extn.lower():
                        print(f"\nFound {file} at\n{parent_folder_path}\n")
                        try:
                            with open(parent_folder_path + "\\" + file, 'r+', encoding='utf8') as hand:
                                text = hand.readlines()
                            print(text)
                            lines.extend(text)
                        # when file cannot be read
                        except UnicodeDecodeError as UDE:
                            print(f"Error when reading {file}\n{UDE}")
                            continue

    # when at least one file was read
    if len(lines) > 0:
        return True, lines

    else:
        print("File Not Found!")
        return False, lines
