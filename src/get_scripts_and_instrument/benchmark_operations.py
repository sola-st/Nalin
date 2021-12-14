"""

Created on 18-March-2020
@author Jibesh Patra

The operations done on dataset of jupyter notebooks
"""
from multiprocessing import Pool, cpu_count
import fileutils as fs
from tqdm import tqdm
import subprocess
from threading import Timer
from zipfile import ZipFile, BadZipFile


def convert_ipynb_to_py(argument):
    time_out_before_killing = 180  # seconds 180 -> 3 minutes
    try:
        def kill_process(p):
            return p.kill()

        p = subprocess.Popen(['jupyter', 'nbconvert', '--to', 'python', argument],
                             stdout=subprocess.PIPE)
        time_out = Timer(time_out_before_killing, kill_process, [p])
        try:
            time_out.start()
            stdout, stderr = p.communicate()
            # print(stdout, stderr)
        finally:
            time_out.cancel()
    except subprocess.TimeoutExpired:
        # print("Timed out")
        pass


def extract_jupyter_notebooks_and_convert_to_py_scripts(in_dir: str, out_dir: str,
                                                        required_number_of_files: int = 1000) -> None:
    """
    Go through the zipped files of in_dir and extract them
    to out_dir. Next, convert the jupyter notebooks to python scripts.

    :param required_number_of_files: The number files to keep
    :param in_dir: The directory that contains zipped jupyter notebooks
    :param out_dir: The directory where the python script will be written
    :return:
    """
    fs.create_dir_list_if_not_present([out_dir])
    zip_files = fs.go_through_dir(directory=in_dir, filter_file_extension='.zip')
    actual_needed_files = required_number_of_files
    for zf in tqdm(zip_files, desc='Going over zips in {}'.format(in_dir), ascii=" #"):
        if required_number_of_files < 1:
            continue
        try:
            with ZipFile(zf, 'r') as zobj:
                file_list = zobj.namelist()
                file_list = file_list[:required_number_of_files]
                for f in file_list:
                    if f.endswith('.ipynb'):
                        # TODO: Check if out_dir already contains a file with the name f and rename it
                        zobj.extract(f, path=out_dir)
                        required_number_of_files -= 1
                        if actual_needed_files and (((
                                                             actual_needed_files - required_number_of_files) / actual_needed_files) * 100) % 10 == 0:
                            print(f"Got {required_number_of_files}/{actual_needed_files} files")
        except BadZipFile:
            print("Can't read zip {}, it is probably broken".format(zf))

    list_of_jupyter_notebooks = fs.go_through_dir(out_dir, '.ipynb')

    # Go over the Jupyter notebooks and convert them to executable python scripts
    with Pool(processes=cpu_count()) as p:
        with tqdm(total=len(list_of_jupyter_notebooks)) as pbar:
            pbar.set_description_str(
                desc="Converting .ipynb files in {} to .py".format(out_dir), refresh=False)
            for i, _ in enumerate(p.imap_unordered(convert_ipynb_to_py, list_of_jupyter_notebooks)):
                pbar.update()
            p.close()
            p.join()

    # Delete the Jupyter notebooks that have been extracted since they are not needed more needed. Instead, we
    # have the corresponding Python scripts.
    with Pool(processes=cpu_count()) as p:
        with tqdm(total=len(list_of_jupyter_notebooks)) as pbar:
            pbar.set_description_str(
                desc="Deleting the original Jupyter notebooks", refresh=False)
            for i, _ in enumerate(p.imap_unordered(delete_file, list_of_jupyter_notebooks)):
                pbar.update()
            p.close()
            p.join()


def delete_file(fl):
    p1 = subprocess.Popen(['rm', fl],
                          stdout=subprocess.PIPE)


if __name__ == '__main__':
    extract_jupyter_notebooks_and_convert_to_py_scripts(in_dir='benchmark/notebooks',
                                                        out_dir='benchmark/notebooks_scripts')
