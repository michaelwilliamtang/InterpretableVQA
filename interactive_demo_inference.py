import yaml
import cv2
import torch
import requests
import numpy as np
import gc
import torch.nn.functional as F
import pandas as pd


import torchvision.models as models
import torchvision.transforms as transforms

from PIL import Image
from IPython.display import display, HTML, clear_output
from ipywidgets import widgets, Layout
from io import BytesIO
from argparse import Namespace


from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.layers import nms
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.utils.model_serialization import load_state_dict


from mmf.datasets.processors.processors import VocabProcessor, VQAAnswerProcessor
from mmf.models.pythia import Pythia
from mmf.common.registry import registry
from mmf.common.sample import Sample, SampleList
from mmf.utils.env import setup_imports
from mmf.utils.configuration import Configuration

setup_imports()

model = PythiaModel()

def init_widgets(url, question):
  image_text = widgets.Text(
    description="Image URL", layout=Layout(minwidth="70%")
  )
  question_text = widgets.Text(
      description="Question", layout=Layout(minwidth="70%")
  )

  image_text.value = url
  question_text.value = question
  submit_button = widgets.Button(description="Ask MMF!")

  display(image_text)
  display(question_text)
  display(submit_button)

  submit_button.on_click(lambda b: on_button_click(
      b, image_text, question_text
  ))

  return image_text, question_text

def on_button_click(b, image_text, question_text):
  clear_output()
  image_path = model.get_actual_image(image_text.value)
  image = Image.open(image_path)

  scores, predictions = model.predict(image_text.value, question_text.value)
  scores = [score * 100 for score in scores]
  df = pd.DataFrame({
      "Prediction": predictions,
      "Confidence": scores
  })

  init_widgets(image_text.value, question_text.value)
  display(image)

  display(HTML(df.to_html()))


image_text, question_text = init_widgets(
    "http://images.cocodataset.org/train2017/000000505539.jpg",
    "where is this place?"
)