import logging
import os

def log(*args):
    print_mess = " ".join([str(arg) for arg in args])
    logging.warning(print_mess + "\n\n")

def create_folder(path):
    if not os.path.exists(path):
        os.mkdir(path)