#!/usr/bin/bash
set -e 

jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to notebook --inplace --execute 03_calculate_properties.ipynb