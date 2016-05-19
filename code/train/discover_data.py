#!/usr/bin/env python3
from unipath import Path, DIRS_NO_LINKS

def find_labels():
    files = {}
    work_dir = Path("tmp/segments")
    if not work_dir.exists():
        raise Exception("You must first create the labels")

    files = {}
    labels = work_dir.listdir(filter=DIRS_NO_LINKS)
    for label in labels:
        files[str(label.name)] = label.listdir(pattern='*.ppm')
    return files

if __name__ == '__main__':
    print(find_labels())
    print(find_labels().keys())
