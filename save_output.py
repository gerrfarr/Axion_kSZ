"""
Python logging routine

author: Gerrit Farren
Last edit: 2020-08-18

Use:
from save_output import SaveOutput

with SaveOutput("path/to/file or opened file") as logger: #will automatically dispose of logger after use and return to standard output stream
    print("whatever")
    #or equivalently
    logger.write("whatever")

"""

import sys
from contextlib import contextmanager
import os
import io


class Logger(io.TextIOWrapper):
    #the logger has the ablity to write to file and standard output stream if stdoutRef is passed
    def __init__(self, file, stdoutRef=None):
        self.terminal = stdoutRef

        if type(file)==str:
            self.log=open(file, 'w')
        elif type(file)==io.TextIOWrapper:
            self.log=file
        else:
            raise ValueError("The parameter file does not have one of the expected types.")

    def write(self, message):
        if self.terminal is not None:
            self.terminal.write(message)
        self.log.write(message)
        self.flush() #if you are concerned about performance and there are a lot of print statements you may want to comment this out and manually flush from time to time

    def close(self):
        self.log.close()

    def flush(self):
        self.log.flush()

@contextmanager
def SaveOutput(target):
    log=Logger(target)
    old_target, sys.stdout = sys.stdout, log # replace sys.stdout
    try:
        yield log # run some code with the replaced stdout
    finally:
        log.close()
        sys.stdout = old_target # restore to the previous value