"""

Created on 21-March-2020
@author Jibesh Patra

Given a folder containing python files execute them using ipython3

"""
import subprocess
from multiprocessing import Pool, cpu_count
from threading import Timer
from tqdm import tqdm
from pathlib import Path
from typing import List, Union
import random
import json, codecs
from typing import List, Any, Union, Dict

random.seed(2)


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
        return []


def writeJSONFile(data: Any, file_path: str) -> Any:
    try:
        # print("Writing JSON file "+file_path)
        json.dump(data, codecs.open(file_path, 'w', encoding='utf-8'),
                  separators=(',', ':'))
    except:
        print("Could not write to " + file_path)


def call_python_execute_one_file(file_path: str) -> Union[None, str]:
    """
    This call is about executing jupyter notebooks.
    :param file_path:
    :return: None if no error encountered while executing the file else the error message
    """

    def kill_process(p):
        p.send_signal(1)
        return p.kill()

    error = None
    time_out_before_killing = 180  # seconds
    try:
        p = subprocess.Popen(['ipython3', '--ipython-dir=/home', file_path],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        time_out = Timer(time_out_before_killing, kill_process, [p])
        try:
            time_out.start()
            stdout, stderr = p.communicate()
            # print(stdout)
            if stderr:
                error = stderr.decode("utf-8")
                # print(error)
        finally:
            time_out.cancel()
    except subprocess.TimeoutExpired:
        error = 'Timeout'
    return error


def execute_files_in_dir(files_in_dir: List) -> None:
    errors_executing_files = []
    with Pool(processes=cpu_count()) as p:
        with tqdm(total=len(files_in_dir)) as pbar:
            pbar.set_description_str(
                desc="Executing using python3", refresh=False)
            for i, err in tqdm(enumerate(p.imap_unordered(call_python_execute_one_file, files_in_dir))):
                if err:
                    errors_executing_files.append((files_in_dir[i], err))
                pbar.update()
            p.close()
            p.join()
    print(
        "\n\tOf {} files, {} encountered errors while executing".format(len(files_in_dir), len(errors_executing_files)))
    writeJSONFile(data=errors_executing_files,
                  file_path='/home/dynamic_analysis_outputs/executing_errors.json')


if __name__ == '__main__':
    if Path('/home/dynamic_analysis_runner/not_executed_files.json').is_file():
        not_executed_files = read_json_file('/home/dynamic_analysis_runner/not_executed_files.json')
        print("Found ", len(not_executed_files),
              'not executed files. Executing them')
        py_files_in_dir = ['/home/python_scripts/' + str(f) for f in not_executed_files]
    else:
        print("Non executed files do not exist")
        py_files_in_dir = list(Path('/home/python_scripts').rglob('*.py'))
    random.shuffle(py_files_in_dir)
    execute_files_in_dir(files_in_dir=py_files_in_dir)
