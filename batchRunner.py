#!/usr/bin/python
import os, sys
import subprocess as sp
import time

files = ['sequences/frame001.png']

if __name__ == "__main__":
    #BatchRunner here is for compressive video sensing setup
    for file in files:
        argv = ['python','runnerGreyScale.py','-f',file]
        sp.Popen(argv)
