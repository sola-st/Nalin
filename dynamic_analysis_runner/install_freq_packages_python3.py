"""

Created on 27-April-2020
@author Jibesh Patra


"""
from collections import Counter
import sys
import subprocess
import json
from typing import List, Any, Union, Dict
import codecs


def read_json_file(json_file_path: str) -> Union[List, Dict]:
    """ Read a JSON file given the file path """
    try:
        obj_text = codecs.open(json_file_path, 'r', encoding='utf-8').read()
        return json.loads(obj_text)
    except FileNotFoundError:
        print(
            "Please provide a correct file p. Eg. ./results/validated-conflicts.json")
        return []
    except Exception as e:
        # Empty JSON file most likely due to abrupt killing of the process while writing
        # print (e)
        return []


def install_a_package(package_name):
    try:
        subprocess.call([sys.executable, '-m', 'pip', 'install', package_name])
    except Exception as e:
        print(f"Could not install {package_name} because {e}")


if __name__ == '__main__':
    # The most frequent packages are obtained using 'src/get_scripts_and_instrument/utils/get_most_frequent_packages.py'
    file_path_to_most_frequent_package = 'most_frequent_packages.json'
    most_frequent_packages = read_json_file(
        json_file_path=file_path_to_most_frequent_package)

    c = Counter(most_frequent_packages)
    K = 100
    top_k = [str(p[0]) for p in c.most_common(K)]

    built_in_packages = sys.builtin_module_names
    print("Will install {} packages".format(len(top_k)))
    for i, pckg in enumerate(top_k):
        print("{}/{} Package: {}".format(i + 1, len(top_k), pckg))
        if pckg not in built_in_packages:
            install_a_package(package_name=pckg)
