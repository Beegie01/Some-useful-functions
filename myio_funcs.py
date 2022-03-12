import joblib
from seaborn.utils import pd, np, plt, os
import seaborn as sns
from sklearn import metrics as mtr, tree, ensemble, model_selection as ms, preprocessing as prep
from tensorflow.keras import backend as K, callbacks, layers, models


def unzip_a_file(filename: str, extract_dir=None, file_password=None):
    """Extract files in a .7z or .zip archive"""
    
    import zipfile
    import py7zr
    import os
    
    if extract_dir is None:
        extract_dir = os.getcwd()
    
    # when file is .zip file
    if zipfile.is_zipfile(filename):
        with open(filename, mode='rb') as f:
            zipfile.ZipFile(f).extractall(path=extract_dir)
    
    # when file is .7z file
    elif py7zr.is_7zfile(filename):
        py7zr.SevenZipFile(filename, password=file_password).extractall(path=extract_dir)
        
    print('Operation Complete.')
    
    
# def file_search(folder_name, search_pattern: str=None, file_ext: str=None):
#     '''
#     returns the full path/location of the file that
#     matches the given search pattern and extension
#     '''
#     if search_pattern is None and file_ext is None:
#         raise ValueError('No valid argument given for search_pattern and file_ext parameter')
#
#     if folder_name is None:
#         raise ValueError('Folder_name to serve as starting point for search is not given')
#
#     result = {}
#     for dirpath, folders, files in os.walk(folder_name):
#
#         for file in files:
#             if search_pattern is not None and file_ext is not None:  # when both params, file_ext and search_pattern, are given
#                 if search_pattern.lower() in file.lower() and f".{file_ext.lower().strip('.')}" in file.lower():
#                     result.setdefault(file, None)
#                     result[file] = f'{dirpath}\\{file}'
#             elif file_ext is not None and search_pattern is None:  # when only param, file_ext, is given
#                 if f".{file_ext.lower().strip('.')}" in file.lower():
#                     result.setdefault(file, None)
#                     result[file] = f'{dirpath}\\{file}'
#             else:  # when only param, search_pattern, is given
#                  if search_pattern.lower() in file.lower():
#                     result.setdefault(file, None)
#                     result[file] = f'{dirpath}\\{file}'
#
#     return result
    
def file_search(search_from: 'path_like_str'=None, search_pattern_in_name: str=None, search_file_type: str=None, print_result: bool=False):
    """
    returns a str containing the full path/location of all the file(s)
    matching the given search pattern and file type
    """
    
    # raise error when invalid arguments are given
    if (search_from is None):
        raise ValueError('Please enter a valid search path')
    if (search_pattern_in_name is None) and (search_file_type is None):
        raise ValueError('Please enter a valid search pattern and/or file type')
    
    search_result = {}
    print(f"Starting search from: {search_from}\n")
    for fpath, folders, files in os.walk(search_from):
        for file in files:
            # when both search pattern and file type are entered
            if (search_file_type is not None) and (search_pattern_in_name is not None):
                if (search_file_type.split('.')[-1].lower() in file.lower().split('.')[-1]) and \
                        (search_pattern_in_name.lower() in file.lower().split('.')[0]):
                    search_result.setdefault(file, f'{fpath}\\{file}')

            # when file type is entered without any search pattern
            elif (search_pattern_in_name is None) and (search_file_type is not None):
                # print(search_file_type)
                if search_file_type.split('.')[-1].lower() in file.lower().split('.')[-1]:
                    search_result.setdefault(file, f'{fpath}\\{file}')    

            # when search pattern is entered without any file type
            elif (search_file_type is None) and (search_pattern_in_name is not None):
                if search_pattern_in_name.lower() in file.lower().split('.')[0]:
                    search_result.setdefault(file, f'{fpath}\\{file}')
                    
    if print_result:
        for k,v in search_result.items():
            print(f"{k.split('.')[0]} is at {v}")
            
    return search_result


def file_search_many(search_from: 'path_like_str' = None,
                 search_pattern_in_names: 'list_or_tuple' = None,
                 search_file_types: 'str_or_list' = None,
                 print_result: bool = False):
    
    if not isinstance(search_pattern_in_names, (tuple, list)):
        raise TypeError("search_pattern_in_names must be a tuple or list")
    if not isinstance(search_file_types, (str, list, tuple)):
        raise TypeError("search_file_types must be a str or list")
                        
    result = dict()
    dtypes_len = len(search_file_types)
                        
    if isinstance(search_file_types, list):
        if dtypes_len == 1:
            search_file_types = search_file_types[0]
        elif (dtypes_len > 1) and (dtypes_len != len(search_pattern_in_names)):
            raise ValueError("search_file_types must have same length with search_pattern_in_names")
                        
    if isinstance(search_file_types, str):                    
        for i in range(len(search_pattern_in_names)):
            result.update(file_search(search_from, search_pattern_in_names[i], search_file_types, print_result))
        return result
                        
    for i in range(len(search_pattern_in_names)):
        result.update(file_search(search_from, search_pattern_in_names[i], search_file_types[i], print_result))
    return result
    
    
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
