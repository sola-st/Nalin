"""

Created on 21-March-2020
@author Jibesh Patra

"""
from pathlib import Path
from AST_transformations import instrument_given_file_multiprocessing, instrument_given_file
import os
from multiprocessing import Pool, cpu_count
from tqdm import tqdm


def should_skip_file(file_path):
    """
    Skip instrumenting files based on some heuristics
    :param file_path:
    :return:
    """
    file_name = Path(file_path).stem
    # Do not want to instrument files like __init__ etc.
    if file_name.startswith('_'):
        return True
    # Do not want to instrument, instrumented files
    if file_name.endswith('_INSTRUMENTED'):
        return True
    return False


def instrument_programs(in_dir, out_dir, out_dir_execution_output, in_place, instrumented_file_suffix):
    """
    Instrument python files in in_dir and write to out_dir
    :param out_dir_execution_output: The directory where the execution outputs of the instrumented files will be written
    :param in_dir:
    :param out_dir:
    :param in_place:
    :param instrumented_file_suffix:
    :return:
    """
    python_files = Path(in_dir).glob('**/*.py')
    files_that_could_not_be_instrumented = []
    print("\nInstrument files from  {} and write to {}".format(in_dir, out_dir))

    files_to_instrument = []
    for file in python_files:
        file_path = str(file)

        # Do not instrument files such as __init__.py etc.
        if should_skip_file(file_path=file_path): continue

        if in_place:
            transformed_file_path = file_path
        else:
            file_name = Path(file_path).stem
            instrumented_file_name = file_name + instrumented_file_suffix
            transformed_file_path = os.path.join(out_dir, instrumented_file_name)
        files_to_instrument.append((file, transformed_file_path, out_dir_execution_output))

    max_ = len(files_to_instrument)
    num_cpu = cpu_count()
    if num_cpu > 5:
        # Multiprocessing support
        with Pool(processes=cpu_count()) as p:
            with tqdm(total=max_) as pbar:
                pbar.set_description_str(
                    desc="Instrumenting files ", refresh=False)
                for i, f in tqdm(
                        enumerate(p.imap_unordered(instrument_given_file_multiprocessing, files_to_instrument))):
                    files_that_could_not_be_instrumented.append(f)
                    pbar.update()
                p.close()
                p.join()
    else:
        for file, transformed_file_path in tqdm(files_to_instrument, desc='Instrumenting files **Sequentially**'):
            tqdm.write(f"Going through {str(file)}")
            f = instrument_given_file(in_file_path=file, out_file_path=transformed_file_path, out_dir_execution_output=out_dir_execution_output)
            files_that_could_not_be_instrumented.append(f)
    print("\n\n\t #### Instrumented {} of {} Python files ####".format(sum(files_that_could_not_be_instrumented),
                                               len(files_that_could_not_be_instrumented)))