import os
from typing import Tuple, List


def get_image_files(image_path, postfix="jpg,png,jpeg"):
    return get_files(dir_path=image_path, postfix=postfix)


def is_posefix(fn: str, postfixes: Tuple[str] | List[str]) -> bool:
    fn = fn.lower()
    base = os.path.basename(fn)
    ext = os.path.splitext(base)[-1]
    # print(fn,base,ext,postfixes)
    if ext in postfixes:
        return True
    return False


def get_files(dir_path, postfix):
    filenames = []
    postfixes = [f".{p.strip().lower()}" for p in postfix.split(",") if len(p.strip()) > 0]
    for root, dirs, files in os.walk(dir_path, topdown=False):
        # print(len(files))
        # print("111", root)
        for fn in files:

            if is_posefix(fn, postfixes):
                # print("22", root)
                filenames.append(os.path.join(root, fn))

    return filenames


if __name__ == "__main__":
    out = "\n".join(get_image_files("../test_images"))
    print(out)
