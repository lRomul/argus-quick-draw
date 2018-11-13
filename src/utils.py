import os


def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def filename_without_ext(file_path):
    basename = os.path.basename(file_path)
    filename = os.path.splitext(basename)[0]
    return filename
