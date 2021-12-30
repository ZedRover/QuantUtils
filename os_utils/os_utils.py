import os 


def file_name_walk(file_dir):
    files = []
    for root, dirs, file in os.walk(file_dir):
        files.extend(file)
    return files 