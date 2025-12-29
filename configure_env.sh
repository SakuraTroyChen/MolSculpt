pip3 install torch-geometric torch-scatter torch-cluster -f https://data.pyg.org/whl/torch-2.5.1+cu124.html
pip3 install setuptools==69.5.1
pip3 install transformers lightning deepspeed rdkit nltk rouge_score peft selfies scikit-learn-extra chardet
pip3 install -U diffusers
pip3 install fcd_torch pomegranate
pip3 install git+https://github.com/molecularsets/moses
pip3 install openbabel-wheel
pip3 install -U deepspeed

# Replace some files
# sudo cp ./sub_files/sascorer.py your-path-to/moses/metrics/SA_Score/sascorer.py
# sudo cp ./sub_files/utils.py your-path-to/moses/metrics/utils.py
