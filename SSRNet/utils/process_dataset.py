"""
UOW, Wed Feb 24 23:37:42 2021
"""
from PIL import Image
import os

# 3. Check a Unicode string
def is_ascii(s):
    return all(ord(c) < 128 for c in s)


# 4. Remove Unicode characters from filenames and save new images. Only apply to the CACD dataset
def process_unicode_filename(in_file, out_file):
    g = open(out_file, "w")
    with open(in_file, "r") as f:
        lines = f.readlines()


        splits = line.split()
        img_path = splits[0]
        img_path_new = img_path.encode('ascii', 'ignore').decode()  # remove the Unicode characters

        # Rename the file
        os.rename(img_path, img_path_new)

                # Write the new filename
                f.write(img_path_new + ' ' + splits[1])
                print(img_path)
            else:
                f.write(line)


# 1. Add a string before each line
def add_string(file_name, string_to_add):
    with open(file_name, 'r') as f:
        file_lines = [''.join([string_to_add, x.strip(), '\n']) for x in f.readlines()]

    with open(file_name, 'w') as f:
        f.writelines(file_lines)


# 2. Remove two last characters (i.e. gender). Only apply to the AADF and AFAD datasaets
def remove_gender(filename):
    with open(filename, "r") as f:
        lines = f.readlines()
    with open(filename, "w") as f:
        for line in lines:
            if line.find("/AAF/") != -1:
                f.write(line[:-2] + "\n")  # remove the last 2 characters (i.e. gender)
            elif line.find("/AFAD/") != -1:
                f.write(line[:-2] + "\n")
            else:
                f.write(line)

