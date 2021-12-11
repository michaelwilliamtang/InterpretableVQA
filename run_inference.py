import pandas as pd
from setup_pythia import PythiaModel

# ignoring warnings
import warnings
warnings.filterwarnings("ignore")

def run(image_text, question_text):
  scores = model.predict_scores(image_text, question_text)
  print(scores)

model = PythiaModel()
run("http://images.cocodataset.org/train2017/000000505539.jpg", "where is this place?")