"""

Created on 21-March-2020
@author Jibesh Patra


This is the starting point for:
 - getting jupyter notebooks
 - converting them to python scripts
 - instrument the python scripts
"""

from transform_programs import instrument_programs
from benchmark_operations import extract_jupyter_notebooks_and_convert_to_py_scripts
import fileutils as fs
from pathlib import Path

def run(python_scripts_dir: str, out_dir_execution_output: str, instrument_in_place: bool = True,
        jupyter_notebooks_dir: str = 'benchmark/jupyter_notebook_datasets') -> None:
    """
    :param python_scripts_dir: The input directory whose python scripts will be instrumented. If the input also includes
                                jupyter notebooks then this is the directory where the .ipynb → .py converted files will
                                be written.
    :param out_dir_execution_output: This is the directory where assignments are serialized when the instrumented files
                                     are executed (for example through Docker container).
                                     Please make sure it exists.
                                     An ABSOLUTE PATH is needed here. A relative path will not work !!
    :param instrument_in_place: A boolean flag representing whether the files instrumented should be overwritten
    :param jupyter_notebooks_dir:  It is expected that the dataset is contained in multiple Zipped files
    :return:
    """
    # This is the directory where the Dynamic Analysis Results will be written
    local_dynamic_analysis_output_dir = f'{str(Path.cwd())}/results/dynamic_analysis_outputs'
    fs.create_dir_list_if_not_present([local_dynamic_analysis_output_dir])

    python_scripts_instrumented_dir = python_scripts_dir + '_instrumented'
    fs.create_dir_list_if_not_present([python_scripts_dir])

    # Create python scripts from jupyter notebooks
    extract_jupyter_notebooks_and_convert_to_py_scripts(in_dir=jupyter_notebooks_dir, out_dir=python_scripts_dir,
                                                required_number_of_files=2000)

    # Instrument python scripts to capture dynamic analysis information
    # If instrument_in_place is True then replace the original file with the instrumented one, else create
    # a new file called _INSTRUMENTED.py
    instrumented_file_suffix = '_INSTRUMENTED.py'
    if instrument_in_place:
        python_scripts_instrumented_dir = python_scripts_dir
    else:
        fs.create_dir_list_if_not_present([python_scripts_instrumented_dir])

    try:
        # Instrument
        instrument_programs(in_dir=python_scripts_dir, out_dir=python_scripts_instrumented_dir,
                            out_dir_execution_output=out_dir_execution_output,
                            in_place=instrument_in_place,
                            instrumented_file_suffix=instrumented_file_suffix)
        print(f"\n\n\t ### You may now execute the instrumented files. The assignments encountered during execution \n\t     will be written to '{local_dynamic_analysis_output_dir}'  ###\n")
    except Exception as e:
        print(e)

if __name__ == '__main__':
    # An absolute path is required for 'out_dir_execution_output'. This path must be present in the Docker 
    # container where the Python files are executed. 
    run(python_scripts_dir='benchmark/python_scripts', out_dir_execution_output='/home/dynamic_analysis_outputs')