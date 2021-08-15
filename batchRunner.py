#!/usr/bin/python
import os, sys
import subprocess as sp
import time

if __name__ == "__main__":
    #BatchRunner here is for compressive video sensing setup
    files = ['sequences/YachtRide.mp4']
    for file in files:
        argv = ['python','runner.py','-f',file]
        sp.Popen(argv)