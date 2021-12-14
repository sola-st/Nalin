"""

Created on 20-March-2020
@author Jibesh Patra

Each Python is instrumented with calls to the 'MAGIC_DYNAMIC_analysis_value_collector' function.
When an assignment is encountered during execution, this function gets
called and the information about the assignment gets stored to the disk.

There are two options for saving the value to disk either using the
.json representation or using the .pickle file. One these options may
be used from the code below.

"""
# from redis.client import Redis

import codecs
import json
import os
import pickle


def writeJSONFile(data, file_path):
    try:
        # print("Writing JSON file "+file_path)
        json.dump(data, codecs.open(file_path, 'w', encoding='utf-8'),
                  separators=(',', ':'))
    except Exception as e:
        pass
        # print("Could not write to " + file_path, e)


def writePickled(data, file_path):
    try:
        with codecs.open(file_path, 'wb') as f:
            pickle.dump(data, f)
    except Exception as e:
        pass


def MAGIC_DYNAMIC_analysis_value_collector(file_name, line_number, var_name, value, outdir):
    extension = ('.json', '.pickle')[1]
    try:
        complete_file_path = file_name
        file_name = os.path.basename(file_name)
        file_name = file_name.split('.')[0]
        try:
            length = len(value)
        except Exception as e:
            length = -1
        default_outdir = '/home/dynamic_analysis_outputs'
        if outdir:
            default_outdir = outdir

        unique_file_name = file_name + '_' + str(line_number) + '_' + var_name + extension
        output_file_path = os.path.join(default_outdir, unique_file_name)
        type_of_variable = str(type(value))
        if extension == '.json':
            writeJSONFile({'file': complete_file_path,
                           'var': var_name,
                           'value': str(value),
                           'line': line_number,
                           'type': type_of_variable,
                           'len': length}, output_file_path)
        else:
            writePickled({'file': complete_file_path,
                          'var': var_name,
                          'value': value,
                          'line': line_number,
                          'type': type_of_variable,
                          'len': length
                          }, output_file_path)
    except Exception as e:
        return
