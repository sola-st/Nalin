"""

Created on 06-April-2020
@author Jibesh Patra

Given a list of python files, get the most common
package that is imported.

We find the most frequent packages and install only top-n from them. This helps in executing many
Python files without dependency errors.
"""
from collections import Counter
from pip._internal import main as pip_main
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import libcst.matchers as matchers
import libcst as cst
from typing import List, Dict, Union, Any
import os
import subprocess
from pathlib import Path
import sys, json, codecs


class GetImports(cst.CSTVisitor):
    def __init__(self):
        super().__init__()
        self.imported_packages = []

    def visit_Import_names(self, node):
        for name in node.names:
            self.imported_packages.append(self.get_name_val(nd=name.name))

    def visit_ImportFrom_names(self, node):
        self.imported_packages.append(self.get_name_val(nd=node.module))

    def get_name_val(self, nd):
        if matchers.matches(nd, matchers.Name()):
            return nd.value
        else:
            return self.get_name_val(nd.value)


def read_file_content(fl: str) -> str:
    with open(fl, 'r') as f:
        content = f.read()
    return content


def all_required_packages_in_a_script(script_path: str) -> List[str]:
    """
    Given path to a script, get the list of all required packages
    :param script_path:
    :return:
    """
    code = read_file_content(script_path)

    try:
        ast = cst.parse_module(source=code)
    except Exception as e:
        ast = []
    try:
        get_imp = GetImports()
        ast.visit(get_imp)
        return get_imp.imported_packages
    except Exception as e:
        return []


def extract_most_frequent_packages(in_dir: str, n: int) -> Dict:
    """
    Go through the python scripts of a given a directory and find the
    imported packages. Get the 'n' most frequent imported package and return
    it
    :param in_dir: a directory containing python files
    :param n: 'n' most frequent imported package
    :return:
    """
    list_of_scripts = list(Path(in_dir).rglob('*.py'))
    package_frequency = Counter()

    with Pool(processes=cpu_count()) as p:
        with tqdm(total=len(list_of_scripts)) as pbar:
            pbar.set_description_str(
                desc="Extracting imported packages ", refresh=False)
            for i, required_packages in tqdm(
                    enumerate(p.imap_unordered(all_required_packages_in_a_script, list_of_scripts))):
                package_frequency.update(required_packages)
                pbar.update()
            p.close()
            p.join()
    # Non multiprocessing
    # for script in tqdm(list_of_scripts, desc='Extracting imported packages'):
    #     required_packages = all_required_packages_in_a_script(script_path=script)
    #     package_frequency.update(required_packages)

    return dict(package_frequency.most_common(n))


def install_a_package(package_name):
    # subprocess.call([sys.executable, '-m', 'pip', 'install', package_name])
    pip_main(['install', package_name])


def install_packages(requirments_file_path):
    subprocess.call([sys.executable, '-m', 'pip',
                     'install', '-r', requirments_file_path])


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


def writeJSONFile(data: Any, file_path: str) -> Any:
    try:
        # print("Writing JSON file "+file_path)
        json.dump(data, codecs.open(file_path, 'w', encoding='utf-8'),
                  separators=(',', ':'), indent=4)
    except:
        print("Could not write to " + file_path)


if __name__ == '__main__':
    # Extract most frequent packages
    working_dir = 'benchmark'
    # print("** Make sure to run 'pip list --format='json' > benchmark/installed.json' before  **")
    # if not fs.pathExists('benchmark/installed.json'):
    # sys.exit(1)

    file_path_to_most_frequent_package = os.path.join(
        working_dir, 'most_frequent_packages.json')

    recompute = 'y'
    if os.path.exists(file_path_to_most_frequent_package):
        recompute = input(
            f"\nThe file containing most frequent packages already exists -> \n'{file_path_to_most_frequent_package}', recompute again 'y'/'n'? \n (Default is 'y') : ")

    if recompute == 'y':
        most_frequent_packages = extract_most_frequent_packages(
            in_dir=os.path.join(working_dir, 'python_scripts'), n=10000)
        writeJSONFile(data=most_frequent_packages, file_path=file_path_to_most_frequent_package)
    else:
        most_frequent_packages = read_json_file(
            json_file_path=file_path_to_most_frequent_package)

    c = Counter(most_frequent_packages)
    print(f"The most common packages are {c.most_common(10)}")
    print(f"\n\tThe output file '{file_path_to_most_frequent_package}'")
