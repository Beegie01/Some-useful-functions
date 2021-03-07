def read_allfile_content(folder_path, file_name=None):
    '''
    copies out the content of a particular file (if file name is given)
    or all files within given folder
    '''

    txt = ""
    FOUND = False

    for parent_folder_path, sub_folders, files in os.walk(folder_path):

        # not interested in any particular file
        if file_name == None:
            for file in files:
                # print(f"\nReading: {file}")
                hand = open(parent_folder_path+"\\"+file)
                txt += hand.read()
                hand.close()

        # searching for a particular file
        else:
            for file in files:
                if file.lower() == file_name.lower():
                    print(f"\nReading: {file}")
                    hand = open(parent_folder_path+"\\"+file)
                    txt += hand.read()
                    hand.close()
                    FOUND = True
                    break

            if FOUND:
                print("Found!!")
                return txt
            else:
                continue

    if len(txt) > 0:
        return txt
    return print("File Not Found!")
