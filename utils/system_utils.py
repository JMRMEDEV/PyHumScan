from os import name, system
import datetime
import time

def clear():
    # for windows
    if name == 'nt':
        _ = system('cls')
    # for mac and linux(here, os.name is 'posix')
    else:
        _ = system('clear')

def get_date():
    # Get current date for file naming
    d = datetime.datetime.now()
    current_date = {'date': d.strftime('%Y_%m_%w'), 'time': d.strftime('%H_%M_%S')}
    return current_date

def get_miliseconds():
    # Get miliseconds based on current time for file naming
    milli_sec = int(round(time.time() * 1000))
    return milli_sec