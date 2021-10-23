class WordCounter(object):

    def __init__(self, text_str_list_or_filename_or_path: list):
        print('This is the Word Counter App!')
        if isinstance(text_str_list_or_filename_or_path, str) and self.textfile_detector(text_str_list_or_filename_or_path):  # read str from textfile
            self.count = self.count_lines(self.read_textfile(text_str_list_or_filename_or_path))
            print(self.count)
        elif isinstance(text_str_list_or_filename_or_path, str) and not self.textfile_detector(text_str_list_or_filename_or_path):  # read str directly
            self.count = self.count_line_words(text_str_list_or_filename_or_path)
            print(self.count)
        elif isinstance(text_str_list_or_filename_or_path, list):  # read str from list of str
            self.count = self.count_lines(text_str_list_or_filename_or_path)
            print(self.count)

    def read_textfile(self, fpath: str) -> list:
        with open(fpath, encoding='utf8') as txt_f:
            return txt_f.readlines()

    def textfile_detector(self, text_str: str) -> bool:
        return ('\\' in text_str ) or '/' in text_str

    def count_line_words(self, line: str, counter: dict=None) -> dict:
        """
        for counting unique words in one string line
        returns a dictionary of each unique word as key
        and the count as value
        :param line:
        :param counter:
        :return:
        """
        line_list = line.lower().split()
        if counter is None:
            counter = dict()
        for ind in range(len(line_list)):
            counter[line_list[ind]] = counter.setdefault(line_list[ind], 0) + 1
            # if line_list[ind].lower() not in counter:
            #     counter[line_list[ind].lower()] = 1
            #     continue
            # counter[line_list[ind].lower()] += 1
        return counter

    def count_lines(self, lines: list) -> dict:
        for ind in range(len(lines)):
            if ind == 0:
                counter = self.count_line_words(lines[ind])
                continue
            counter = self.count_line_words(lines[ind], counter)
        return {k: v for v,k in sorted([(v,k) for k,v in counter.items()], reverse=True)}


if __name__ == '__main__':
    txt = input("Enter text string or text file location:\t")
    WordCounter(txt)
