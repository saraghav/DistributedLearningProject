#!/usr/bin/python3
import logging
import logging.handlers
import os
import sys

def setup_root_logger(logfilename, logginglevel):
    """ sets up root logger
    """
    logging.getLogger('').setLevel(logginglevel)

    # set a format that's desired
    formatter = logging.Formatter(fmt='%(asctime)s %(name)s %(levelname)s:\t%(message)s', datefmt='%I:%M:%S %p')

    # define a Handler which writes INFO messages or higher to the sys.stdout
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logginglevel)
    # tell the handler to use this format
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)

    # define a rotating file handler to preseve older log files
    logfilename = os.path.basename(logfilename)
    logfile = logging.handlers.RotatingFileHandler(filename=logfilename, mode='a', backupCount=10)
    logfile.setLevel(logginglevel)
    # tell the handler to use the defined format
    logfile.setFormatter(formatter)
    # do rollover if logfile exists
    if os.path.isfile(logfilename):
        logfile.doRollover()
    # add the handler to the root logger
    logging.getLogger('').addHandler(logfile)

def setup_logger(loggername):
    """ sets up module specific logger.
    returns: logger instance
    """
    logger = logging.getLogger(loggername)
    return logger
