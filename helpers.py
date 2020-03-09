from pathos.parallel import ParallelPool
from pathos.multiprocessing import ProcessPool
import numpy as np
import pathos
from contextlib import contextmanager
import multiprocess
import subprocess
import shlex
import sys


class NoDaemonProcess(multiprocess.Process):
    def _get_daemon(self):
        return False

    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)


class NoDaemonPoolClass(pathos.multiprocessing.Pool):
    def Process(self, *args, **kwds):
        return NoDaemonProcess(*args, **kwds)


class PseudoPool():
    def __init__(self):
        return None

    def imap(self, func, vals):
            return map(func, vals)

    def map(self, func, vals):
            return map(func, vals)

    def amap(self, func, vals):
            return map(func, vals)


@contextmanager
def MyProcessPool(nodes=None):
    if nodes is None or nodes>1:
        p=ProcessPool(nodes)
        try:
            yield p
        finally:
            p.close()
            p.join()
            p.clear()
    else:
        #print("Using PseudoPool!")
        yield PseudoPool()


@contextmanager
def MyParallelPool(nodes=None):
    if nodes is None or nodes>1:
        p=ParallelPool(nodes)
        try:
            yield p
        finally:
            p.close()
            p.join()
            p.clear()
    else:
        #print("Using PseudoPool!")
        yield PseudoPool()


@contextmanager
def NoDaemonPool(processes=None):
    p=NoDaemonPoolClass(processes)
    try:
        yield p
    finally:
        p.close()
        p.join()
        p.clear()


def execute(cmd):
    popen = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    for stdout_line in iter(popen.stdout.readline, ""):
        yield stdout_line
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)


def is_array(item, cl=None):
    try:
        if len(item)>1 and (cl is None or isinstance(item, cl)):
            return True
        else:
            return True
    except TypeError as ex:
        return False
    except AttributeError as ex:
        return False


# TEST and implement
def reconvertMesh(mesh):
    if np.allclose(mesh[:,0], mesh[0,0]):
        return mesh[0,:]
    elif np.allclose(mesh[0,:], mesh[0,0]):
        return mesh[:,0]
    else:
        raise Exception("The given array is not a meshgrid.")
