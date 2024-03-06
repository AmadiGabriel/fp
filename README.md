# Fault Prediction for Operational Aero-Engines: A Case Study with a Data-driven Approach

## Description

This program performs the full model building procedure for Problem FPs with LightGBM using the features computed in the case study. 
The program has the option of building the model using the true labels, thus recovering the cross-validation performance shown in Table 4. 
The program also has the option of building the model from one instance of shuffled labels, thus producing a random cross-validation performance
in line with the histogram of Figure 6.

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


## Usage

git clone https://github.com/amadiudu/fp.git
cd fp
pip install -r requirements.txt
python main.py


## Libraries

The following libraries are used in the Python file:
- joblib
- pandas
- numpy
- scikit-learn
- lightgbm
- warnings

## File Structure

- The file to run is `main.py`.
- The dependency file for class mislabelling is located in the `dependency` folder named `mislabel.py`.
- The engine data are located in the folder `engine_data`.

## Research Paper

This project supports the research paper titled "Fault Prediction for Operational Aero-Engines: A Case Study with a Data-driven Approach".

## Authors

- Amadi Gabriel Udua
  - School of Engineering, University of Leicester, University Road, Leicester LE1 7RH, UK
  - Air Force Institute of Technology, PMB 2014, Kaduna, Nigeria
- Andrea Lecchini-Visintinic
  - School of Electronics and Computer Science, University of Southampton, University Road, Southampton SO17 1BJ, UK
