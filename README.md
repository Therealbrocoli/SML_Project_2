# SML Project 2
This folder contains the student template of project 2


## Environment Setup
### GPU Cluster
We have prepared an environment on the GPU cluster ran by the IT support group of the computer science department.
See the project information sheet (and the links therein) for more information.

### Google Colab
For those who are running the project with Google Colab, we prepared the interface to run the code in the `Instructions_GoogleColab.ipynb`.

Please check out the installation guide on Moodle for this.
Make sure you upload the whole project2 folder (including the uncompressed datasets) to your Google Drive and follow the instructions in the `Instructions_GoogleColab.ipynb` to run the code.

### Local Installation
If you have a computer with a GPU, you might want to run the project locally. In this case, please set up an Anaconda environment running `python3.10`. Please check out the installation guide, e.g. from [Informatik II](https://lec.inf.ethz.ch/mavt/informatik2/2024/exercises/exercise1.pdf) for this.

If you are using Windows, we recommend to use either the VS code terminal or the Anaconda terminal, which is installed with Anaconda.

########################################################################################################
Please activate your project 2 environment by using:
```
conda activate <environment_name>
```
Then navigate to the folder containing the project files and run:
```
pip install --upgrade pip
pip install -r requirements.txt
```
If you require any additional packages, run:
```
pip install <package_name>
```
Then please run
```
pip freeze > requirements.txt
```
To save the current pip entries to your pip
````####################################################################################################
Ihr müsst das env selber erstellen, am besten geht ihr in die anaconda powershell schreibt dort

conda init powershell

Dann könnt ihr conda als Begriff in VS benutzen

conda create —name project2_env python=3.11

Jetzt habt ihr ein globales Environment erstellt.

Jetzt könnt ihr in VS gehen, falls ihr es noch nicht seit, schauendes der Ordnerpfad stimmt
Dann 
conda activate project2_env
python.exe -m pip install —upgrade pip
pip install -r requirements.txt
``
#######################################################################################################
Make sure to extract all the data in the `./datasets` folder.

## Running Code
Please note that the script `train.py` takes arguments when you run them. These arguments are used when the script is carried out. The arguments to a python script can be specified in the following manner:
```bash
python train.py --<argument_1_name> <argument_1_value> --<argument_2_name> <argument_2_value>
```
`train.py` takes two arguments, namely the path to the datasets folder and where the training loop should save your model checkpoints.

For more information on the available arguments to these scripts, please run the following command:
```bash
python train.py -h
```

### GPU Cluster
See the instructions on the project document and the example in the file `my_job.sh`

### Google Colab
Check out the `Instructions_GoogleColab.ipynb` for instructions on how to run the code in Google Colab.

### Local Installation
To run your solution locally, first make sure you have activated your conda environment. Then open a terminal and run the following command with your arguments to train the model:
```bash
python train.py <your_arguments_here>
```
