"""

Created on 28-October-2020
@author Jibesh Patra


"""
import os
from run_get_scripts_and_instrument import run as instrument_dir

repo_dir_root = 'benchmark/python_repos'
dynamic_analysis_output_results_dir = 'results/test_dynamic_analysis_outputs'

repo_list_dir = os.listdir(repo_dir_root)

for repo_dir in repo_list_dir:
    complete_dir_path = os.path.join(repo_dir_root, repo_dir)
    # Absolute path is needed
    instrument_dir(python_scripts_dir=complete_dir_path,
                   out_dir_execution_output=f'/home/jibesh/nn-dynamic-analysis-bug-finder/{dynamic_analysis_output_results_dir}/{repo_dir}')
