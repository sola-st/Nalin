🌸 Nalin: Learning from Runtime Behavior to Find Name-Value Inconsistencies in Jupyter Notebooks
---

There are two main components of the approach:

1. Obtain runtime data using _dynamic analysis_. The data here is assignments encountered during execution.
2. Train a classifier and find inconsistencies.

## Requirements & Setup

**Directory Structure**

The directory structure is as follows:

```shell
src/ # The root directory of all source files
benchmark/ # This my contain the input Python files
src/dynamic_analysis_tracker_local_package # Code for saving the assignments encountered during dynamic analysis
src/get_scripts_and_instrument # Code for running dynamic analysis
src/nn # Code for running the Neural Network and creating Negative examples
src/view_probable_bugs # Use a browser to view assignment locations flagged by the classifier as buggy
results # The results generated by running the experiments are written here
```

**Python & Packages**

Python version 3.8 may be used and the required packages are listed in _requirements.txt_. The packages may be installed using the command _pip install -r requirements.txt_.

**Jupyter Notebook Dataset**

We use the dataset from a [CHI’18](https://dl.acm.org/doi/10.1145/3173574.3173606) paper that has analyzed more than 1.3 million publicly available Jupyter 
Notebooks from GitHub. Download the dataset using the [link](https://library.ucsd.edu/dc/collection/bb6931851t).

---

## 1. Dynamic Analysis ⚙️


### How to execute Jupyter notebooks from command line?

We want to execute a large number of Jupyter notebooks. We follow the following steps:

- Convert the Jupyter notebooks to Python scripts.
- Instrument the Python scripts individually.
- Execute the instrumented scripts.
- Collect the run-time data as:
    - JSON files where a string representation of the data is saved. _OR_
    - Pickled files where the value is stored in a binary format that may be read later. WARNING: Takes a lot of disk
      space.

### Instrument Python files for tracking assignments

The directory is _src/get_scripts_and_instrument_

Run the following command from the root folder.

```bash
python src/get_scripts_and_instrument/run_get_scripts_and_instrument.py
```

### Execute the instrumented Python files in a Docker container

In many occasions, it has been found that the files being executed makes unsolicited network requests and downloads
large datasets. This can lead to filling up the disk space quickly. We avoid this completely by running the instrumented
python files in a docker container. More specifically, we execute the instrumented files using _ipython3_. Executing
each file generates many JSON/pickle files if there exists any assignments which are in scope (We do not track
assignments of type a.b.c = m or a\[b] = c or aug-assignments of type a\[b]+=2). Each generated file correspond to an
assignment.

**The Dockerfile**

Save it as _Dockerfile_ with no extension at the root directory of the project. Notice the last line of the Dockerfile.
When docker gets executed, this is the command that is run.

```dockerfile
FROM python:3.7.5
COPY src/dynamic_analysis_tracker_local_package /home/dynamic_analysis_tracker_local_package

WORKDIR /home

RUN python3 -m pip install -e dynamic_analysis_tracker_local_package
RUN python3 -m pip install --upgrade pip
# Install some required packages
RUN python3 -m pip install \
		tqdm \
		jupyter \
		ipython
# Install the most frequent packages
COPY dynamic_analysis_runner/most_frequent_packages.json /home
COPY dynamic_analysis_runner/install_freq_packages_python3.py /home
COPY dynamic_analysis_runner/fileutils.py /home
RUN python3 install_freq_packages_python3.py

# Create the mount points to be used during running the docker container
RUN mkdir -p /home/dynamic_analysis_runner
# We mount the scripts that we want to execute at this path
RUN mkdir -p /home/python_scripts
# The output of the dynamic analysis
RUN mkdir -p /home/dynamic_analysis_outputs
# The default profile folder for ipython3
RUN mkdir -p /home/profile_default
# RUN ls -al > /home/dynamic_analysis_outputs/directories_in_docker.txt

# Create a working directory
RUN mkdir -p /home/temp_working_dir_inside_docker
WORKDIR /home/temp_working_dir_inside_docker

CMD python3 /home/dynamic_analysis_runner/execute_files_py3.py
```

**The general structure is something like the following:**

We mount two folders (read only). One contains own scripts while the other contains the Python files we want to
execute. In addition, we also mount another folder (writable) where the data is written by the executing scripts.

- Build the docker image using the following command from the root of the project directory:

```bash
sudo docker build -t dynamic_analysis .
```

- Run the Docker image (Python3)

```bash
sudo docker run --network none -v "$(pwd)"/dynamic_analysis_runner:/home/dynamic_analysis_runner:ro -v "$(pwd)"/benchmark/python_scripts:/home/python_scripts:ro  -v "$(pwd)"/results/dynamic_analysis_outputs:/home/dynamic_analysis_outputs -v "$(pwd)"/profile_default:/home/profile_default:Z -it --rm dynamic-analysis-py3
```

On running the above command, it should write the dynamic analysis results to the directory _results/dynamic_analysis_outputs_. 

---

## 2. Classifier 🧐

Hopefully, the dynamic analysis step generated many JSON/Pickle files. Each generated file represents one assignment (
Eg. num=calc(1,2)) encountered during execution. The next step is to put the individual assignment files together and create a single
file. We call this file _positive_examples_.

To be effective, a classifier needs both positive and negative examples. We have two ways of create negative examples.
_One_ is to generate them randomly while the 11 others use some heuristic for generating them. Both ways of generating
negative examples explained later.

All experiments using the classifier is run using the command **python src/nn/run_classification.py**.

### Pre-Processing and Creating Negative Examples

All pre-processing and the creation of the negative examples happens at the _process()_ call of run_classification.py.
You may refer to the documentation of _process_ to understand how it works.

The extracted positive examples need to be pre-processed before being used for generating negative examples. We use
different heuristics for the pre-processing step. Refer to the code for more details.

### Run training

````bash
python src/nn/run_classification.py --train --num-epochs=5 --name='RNNClassifier'
````

### Run testing

````shell
python src/nn/run_classification.py --test --name=_p --saved-model=results/saved_models/RNNClassifier.pt --test-dataset=results/test_examples.pkl
````