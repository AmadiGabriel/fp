# Fault Prediction for Operational Aero-Engines: A Case Study with a Data-Driven Approach

## Description

This program performs the full model building procedure for Problem FPs with LightGBM using the features computed in the case study. The program has the option of building the model using the true labels, thus recovering the cross-validation performance shown in Table 4. The program also has the option of building the model from instances of shuffled labels, thus producing random cross-validation performances in line with the histogram of Figure 7.

## Table of Contents

1. [Installation](#installation)
2. [Usage](#usage)
3. [Libraries](#libraries)
4. [File Structure](#file-structure)
5. [Research Paper](#research-paper)
6. [Authors](#authors)

## Installation

## Installation

We recommend setting up a virtual environment before installing the project dependencies. 
Use the following commands to create and activate a virtual environment:

```bash
python3 -m venv myenv
source myenv/bin/activate  # For Unix/Linux
.\myenv\Scripts\activate   # For Windows
```

## Usage
```bash
git clone https://github.com/AmadiGabriel/fp.git
cd fp
pip install -r requirements.txt
python3 main.py
```
In main.py:
Line 1, Set 
```bash
permute = True
```
to build the model of shuffled labels.


## Libraries

The following libraries are used in the Python file:
- joblib
- pandas
- numpy
- scikit-learn
- lightgbm
- warnings
- tqdm

## File Structure

- The file to run is `main.py`.
- A dependency file to show the progress of the execution of the main.py is located in the `dependency` folder named `progress.py`. This function called when the program is run.
- The engine data are located in the folder `engine_data`. The files are called when the program is run.

## Research Paper

This project supports the research paper titled "Fault Prediction for Operational Aero-Engines: A Case Study with a Data-driven Approach".

## Authors

- Amadi Gabriel Udu
  - School of Engineering, University of Leicester, University Road, Leicester LE1 7RH, UK
  - Air Force Institute of Technology, PMB 2014, Kaduna, Nigeria
- Andrea Lecchini-Visintini
  - School of Electronics and Computer Science, University of Southampton, University Road, Southampton SO17 1BJ, UK
