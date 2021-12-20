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