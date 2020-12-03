# Simple Baseline for Cinc2017 ECG Classification Task 

## Before Running
First, you should clone the repo.
```
git clone https://github.com/King-HAW/ECG_Classification_Baseline
```

You should install all the dependencies by running:
```
cd ./ECG_Classification_Baseline
pip install -r requirements.txt
```

You can run the following command to explore the data distribution.
```
cd ./get_meta
python get_meta.py
```

Also, you should running the following command to do the prepeocess. You can get some files after you run the command.
**trainset.npy** and **validset.npy** are data files.
**training-nodup.csv** and **infer.csv** are index files.

```
cd ./preprocess
python preprocessing.py
```

## Training

Just run the **train.py**, the trainset will be divided into five folds and cross-validation will run automatically. You can change some hyperparameters pre-defined in the script to fit your environment.
```
python train.py
```

## Testing

After the training step is done, run the **predict.py** to get the 5-fold ensemble result.
```
python predict.py
```

## Tips
You are encouraged to change the code to get better performance. Maybe the self-attention module is a good choice. Have fun and hope you can enjoy it :-)
