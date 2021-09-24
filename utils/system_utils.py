from os import name, system
import time

def clear():
    # for windows
    if name == 'nt':
        _ = system('cls')
    # for mac and linux(here, os.name is 'posix')
    else:
        _ = system('clear')

def get_date():
    # Get current time for file naming
    t = time.localtime()
    current_time = time.strftime("%H_%M_%S", t)
    return current_time

def get_miliseconds():
    # Get miliseconds based on current time for file naming
    milli_sec = int(round(time.time() * 1000))
    return milli_sec