"""

Created on 16-November-2020
@author Jibesh Patra

Use the prediction as input to copy the original python files where there exists a warning.
"""
import shutil
import os
def create_dir_list_if_not_present(paths) -> None:
    # Check of type of paths is list
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)

file_names_all_types = [
    "ipynb_scripts/nb_502423.py",
    "ipynb_scripts/nb_1024433.py",
    "ipynb_scripts/nb_852765.py",
    "ipynb_scripts/nb_233164.py",
    "ipynb_scripts/nb_1201093.py",
    "ipynb_scripts/nb_490046.py",
    "ipynb_scripts/nb_347443.py",
    "ipynb_scripts/nb_234684.py",
    "ipynb_scripts/nb_1240239.py",
    "ipynb_scripts/nb_912169.py",
    "ipynb_scripts/nb_854745.py",
    "ipynb_scripts/nb_238184.py",
    "ipynb_scripts/nb_277365.py",
    "ipynb_scripts/nb_502423.py",
    "ipynb_scripts/nb_827818.py",
    "ipynb_scripts/nb_499105.py",
    "ipynb_scripts/nb_597139.py",
    "ipynb_scripts/nb_440265.py",
    "ipynb_scripts/nb_798594.py",
    "ipynb_scripts/nb_637947.py",
    "ipynb_scripts/nb_368393.py",
    "ipynb_scripts/nb_1077269.py",
    "ipynb_scripts/nb_423186.py",
    "ipynb_scripts/nb_702013.py",
    "ipynb_scripts/nb_708089.py",
    "ipynb_scripts/nb_986269.py",
    "ipynb_scripts/nb_423186.py",
    "ipynb_scripts/nb_1178833.py",
    "ipynb_scripts/nb_501068.py",
    "ipynb_scripts/nb_1125553.py"]

dest_dir = '../../results/DeepBugsPlugin_Inspection'
create_dir_list_if_not_present([dest_dir])

for fl in file_names_all_types:
    dest_link = f'{dest_dir}/{fl}'
    fl = f'../view_probable_bugs/{fl}'
    shutil.copyfile(fl, dest_link)
