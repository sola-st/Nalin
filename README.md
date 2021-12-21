🌸 Nalin: Learning from Runtime Behavior to Find Name-Value Inconsistencies in Jupyter Notebooks
---

There are two main components of the approach:

1. Obtain runtime data using _dynamic analysis_. The data here is assignments encountered during execution.
2. Train a classifier and find inconsistencies.

## TL;DR 🪜

Simply run or read the 📌 marked commands from the root directory. 

## Requirements & Setup

We have tested using Ubuntu 18.04 LTS and Python 3.8.12

**Directory Structure**

The directory structure is as follows:

```shell
src/ # The root directory of all source files
benchmark/ # This may contain the input Python files & the Jupyter Notebooks
dynamic_analysis_runner # Code for running Dynamic Analysis
src/dynamic_analysis_tracker_local_package # Python package for saving the assignments encountered during execution
src/get_scripts_and_instrument # Code for getting Jupyter Notebooks, converting them to Python scripts and instrumenting
src/nn # Code for running the Neural Network 
src/view_probable_bugs # Use a browser to view assignment locations flagged by the classifier as buggy
results # The results generated by running the experiments are written here
```

**Python & Packages**

The required packages are listed in _requirements.txt_. The packages may be installed using the command _pip install -r requirements.txt_. Additionally install the [PyTorch](https://pytorch.org/get-started/locally/) package (We have tested on PyTorch version 1.10.1).  

📌
```shell
pip install -r requirements.txt
pip install torch==1.10.1+cpu torchvision==0.11.2+cpu torchaudio==0.10.1+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html
```

💡 The above command will install the CPU version of PyTorch. If you want CUDA support, please change the command accordingly as mentioned in the [link](https://pytorch.org/get-started/locally).

**Jupyter Notebook Dataset**

We use the dataset from a [CHI’18](https://dl.acm.org/doi/10.1145/3173574.3173606) paper that has analyzed more than 1.3 million publicly available Jupyter 
Notebooks from GitHub. Download the dataset using the [link](https://library.ucsd.edu/dc/collection/bb6931851t).
We provide a sample of 100 Jupyter notebooks (_benchmark/jupyter_notebook_datasets/sample.zip_) obtained from this dataset for testing. 

**Embedding**

📌
Download the embedding file from the [link](https://u.pcloud.link/publink/show?code=XZyeJaXZrnrbvwzBcYSOWYgzsn4usJ6DOqPy) and put in the _benchmark_ folder.

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

📌
```bash
python src/get_scripts_and_instrument/run_get_scripts_and_instrument.py
```

By default, this script should: 
1) Extract the Jupyter notebooks present in '_benchmark/jupyter_notebook_datasets/sample.zip_' to '_benchmark/python_scripts_'
2) Convert the extracted notebooks to Python script 
3) Delete the extracted notebooks
4) Instrument the converted Python scripts

Not all Jupyter Notebooks present in _sample.zip_ get instrumented. Some encounter errors while conversion to Python
scripts and some during instrumentation. 

### Execute the instrumented Python files in a Docker container

In many occasions, it has been found that the files being executed makes unsolicited network requests and downloads
large datasets. This can lead to filling up the disk space quickly. We avoid this completely by running the instrumented
python files in a docker container. More specifically, we execute the instrumented files using _ipython3_. Executing
each file generates many JSON/pickle files if there exists any assignments which are in scope (We do not track
assignments of type a.b.c = m or a\[b] = c or aug-assignments of type a\[b]+=2). Each generated file correspond to an
assignment.

**Dockerfile**

The following Dockerfile is included in root directory. 
Notice the last line of the Dockerfile. When docker gets executed, this is the command that is run.

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
RUN python3 install_freq_packages_python3.py

# Create the Directories that will be mounted during Running the docker container
RUN mkdir -p /home/dynamic_analysis_runner
# We will mount the scripts that we want to execute here
RUN mkdir -p /home/python_scripts
# We will mount when 
RUN mkdir -p /home/dynamic_analysis_outputs
RUN mkdir -p /home/profile_default

# Create a working directory
RUN mkdir -p /home/temp_working_dir_inside_docker

# For debugging, check if all directories have been created properly
# RUN ls -al > /home/dynamic_analysis_outputs/directories_in_docker.txt
WORKDIR /home/temp_working_dir_inside_docker


CMD python3 /home/dynamic_analysis_runner/execute_files_py3.py
```

**The general structure is something like the following:**

Build the docker image using the following command from the root of the project directory:

📌
```shell
sudo docker build -t nalin_dynamic_analysis_runner .
```

On building the image, it should create the required folders in the Docker container. Some of these 
folders are useful for mounting required local folders (eg. folder containing the instrumented Python scripts) while running the image.

Additionally, it should also install the most common 100 packages present in _benchmark/python_scripts_. This may be obtained by running
```python src/get_scripts_and_instrument/utils/get_most_frequent_packages.py```. If you do not want to re-run, we provide a pre-computed 
file at _dynamic_analysis_runner/most_frequent_packages.json_.

Run the Docker image:

📌
```bash
sudo docker run --network none \
-v "$(pwd)"/dynamic_analysis_runner:/home/dynamic_analysis_runner:ro \
-v "$(pwd)"/benchmark/python_scripts:/home/python_scripts:ro \
-v "$(pwd)"/results/dynamic_analysis_outputs:/home/dynamic_analysis_outputs \
-v "$(pwd)"/benchmark/profile_default:/home/profile_default:Z \
-it --rm nalin_dynamic_analysis_runner
```

On running the above command, it should write the dynamic analysis results to the directory _results/dynamic_analysis_outputs_.

We mount two folders (read only). One contains own scripts while the other contains the Python files we want to
execute. 
  - The _dynamic_analysis_runner_ folder gets mounted at the home directory of the Docker container.
  - The _benchmark/python_scripts_ that contain the instrumented Python files also gets mounted at the home directory of the Docker container.

In addition, we also mount another folder (writable) where the data is written by the executing scripts.
  - The _/results/dynamic_analysis_outputs_ folder gets mounted at the home directory of the Docker container.

The _benchmark/profile_default_ folder and its content need to be mounted to avoid some iPython specific errors. 


**Dynamic Analysis Output**

By default, the dynamic analysis outputs are written to the _'results/dynamic_analysis_outputs'_. Make sure
this path exists. 

---

## 2. Classifier 🧐

Hopefully, the dynamic analysis step generated many JSON/Pickle files. Each generated file represents one assignment (
Eg. num=calc(1,2)) encountered during execution. The next step is to put the individual assignment files together and create a single
file. We call this file _positive_examples_.

To be effective, a classifier needs both positive and negative examples. We have two ways of create negative examples.
One is to generate them randomly while the other use some heuristic for generating them. 

All experiments using the classifier is run using the command **python src/nn/run_classification.py**.

### Pre-Processing and Creating Negative Examples

All pre-processing and the creation of the negative examples happens at the _process()_ call of run_classification.py.
You may refer to the documentation of _process_ to understand how it works.

The extracted positive examples need to be pre-processed before being used for generating negative examples. We use
different heuristics for the pre-processing step. Refer to the code for more details.

### Run training

📌
````bash
python src/nn/run_classification.py --train --num-epochs=5 --name='Nalin'
````

### Run testing

📌
````shell
python src/nn/run_classification.py --test --saved-model=results/saved_models/RNNClassifier_Nalin.pt --test-dataset=results/test_examples.pkl
````