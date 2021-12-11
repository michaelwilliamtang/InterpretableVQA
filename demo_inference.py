import pandas as pd
from setup_pythia import PythiaModel

# ignoring warnings
import warnings
warnings.filterwarnings("ignore")

def run(image_text, question_text):
  scores, predictions = model.predict(image_text, question_text)
  scores = [score * 100 for score in scores]
  df = pd.DataFrame({
      "prediction": predictions,
      "confidence": scores
  })
  print(df)

model = PythiaModel()
run("http://images.cocodataset.org/train2017/000000505539.jpg", "where is this place?")