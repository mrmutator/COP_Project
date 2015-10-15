__author__ = 'rwechsler'
import glob
from classification import evaluate_model

for f in glob.glob("models/*.model"):
    fname = f.split(".")[0]
    evaluate_model(fname)