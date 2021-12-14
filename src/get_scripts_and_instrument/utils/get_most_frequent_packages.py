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
import fileutils as fs
from typing import List, Dict
import os
import subprocess
import sys

sys.path.extend(['src'])


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


def all_required_packages_in_a_script(script_path: str) -> List[str]:
    """
    Given path to a script, get the list of all required packages
    :param script_path:
    :return:
    """
    code = fs.read_file_content(script_path)

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
    list_of_scripts = fs.go_through_dir(
        directory=in_dir, filter_file_extension='.py')
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


if __name__ == '__main__':
    # Extract most frequent packages and install them
    working_dir = 'benchmark'
    # print("** Make sure to run 'pip list --format='json' > benchmark/installed.json' before  **")
    # if not fs.pathExists('benchmark/installed.json'):
    # sys.exit(1)

    file_path_to_most_frequent_package = os.path.join(
        working_dir, 'most_frequent_packages.json')

    recompute = 'y'
    if fs.pathExists(file_path_to_most_frequent_package):
        recompute = input(
            "Most frequent packages already exists, recompute again y/n?")

    if recompute == 'y':
        most_frequent_packages = extract_most_frequent_packages(
            in_dir=os.path.join(working_dir, 'notebooks'), n=10000)
        fs.writeJSONFile(data=most_frequent_packages,
                         file_path=file_path_to_most_frequent_package)
    else:
        most_frequent_packages = fs.read_json_file(
            json_file_path=file_path_to_most_frequent_package)

    c = Counter(most_frequent_packages)
    print(f"The most common packages are {c.most_common(10)}")
