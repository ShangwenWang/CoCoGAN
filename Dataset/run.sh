unzip dataset.zip
cd dataset
wget https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/python.zip
wget https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/java.zip

unzip python.zip
unzip java.zip
rm *.zip
rm *.pkl

python preprocess.py
rm -r */final
cd ..