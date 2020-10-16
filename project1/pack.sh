#!/bin/bash

rm -r data/plots
cd data
python3 dt.py
python3 knn.py
python3 residual_fitting.py
cd ..

ls -R data/plots

pdflatex elen0062.tex # > /tmp/log
while grep 'Rerun to get ' /tmp/log ; do pdflatex elen0062.tex  > /tmp/log ; done


tar czf elen0062-prj1.tar.gz elen0062.pdf data/*.py
