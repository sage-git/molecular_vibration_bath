#!/usr/bin/python

import os
import sys
import subprocess
from datetime import datetime
from subprocess import PIPE

def get_current_env():
    current_commit = subprocess.run(["git", "rev-parse", "HEAD"], stdout=PIPE)
    current_scomm = subprocess.run(["git", "rev-parse", "--short", "HEAD"], stdout=PIPE)
    current_diff = subprocess.run(["git", "diff", "HEAD"], stdout=PIPE)
    current_desc = subprocess.run(["git", "describe", "--all"], stdout=PIPE)
    ret = "Executing command:\n"
    ret += "{}\n".format(" ".join(sys.argv))
    ret += "\n"
    ret += "Execution date:\n"
    ret += "{}\n".format(datetime.now())
    ret += "\n"
    ret += "Execution host:\n"
    ret += "{}\n".format(os.uname()[1])
    ret += "\n"
    ret += "Execution pwd:\n"
    ret += "{}\n".format(os.getcwd())
    ret += "\n"
    ret += "git commit:\n"
    ret += "{}".format(current_commit.stdout.decode("utf-8"))
    ret += "short:\n"
    ret += "{}".format(current_scomm.stdout.decode("utf-8"))
    ret += "describe:\n"
    ret += "{}".format(current_desc.stdout.decode("utf-8"))
    ret += "\n"
    ret += "git diff:\n"
    ret += "{}".format(current_diff.stdout.decode("utf-8"))
    return ret

if __name__ == "__main__":
    print(get_current_env())

