import matplotlib.pyplot as plt
import torch
from torch.nn.functional import softmax, interpolate
from torchvision.io.image import read_image
from torchvision.models import resnet18
from torchvision.transforms.functional import normalize, resize, to_pil_image

from torchcam.methods import SmoothGradCAMpp, LayerCAM
from torchcam.utils import overlay_mask

from setup_pythia import PythiaModel

# ignoring warnings
import warnings
warnings.filterwarnings("ignore")

def grad_cam(model, image_path, question_text):
    cam_extractor = SmoothGradCAMpp(model)
    img = read_image(image_path)
    # Preprocess it for your chosen model
    input_tensor = normalize(resize(img, (224, 224)) / 255., [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    # Preprocess your data and feed it to the model
    out = model(input_tensor.unsqueeze(0))
    # Retrieve the CAM by passing the class index and the model output
    cams = cam_extractor(out.squeeze(0).argmax().item(), out)

    # Notice that there is one CAM per target layer (here only 1)
    for cam in cams:
        print(cam.shape)

    # The raw CAM
    for name, cam in zip(cam_extractor.target_names, cams):
        plt.imshow(cam.numpy())
        plt.axis('off')
        plt.title(name)
        plt.show()

    # Overlayed on the image
    for name, cam in zip(cam_extractor.target_names, cams):
        result = overlay_mask(to_pil_image(img), to_pil_image(cam, mode='F'), alpha=0.5)
        plt.imshow(result)
        plt.axis('off')
        plt.title(name)
        plt.show()

    # Once you're finished, clear the hooks on your model
        cam_extractor.clear_hooks()


# model = PythiaModel().get_pythia_model()
model = PythiaModel().get_resnet_model()
grad_cam(model, "border-collie.jpg", "where is this place?")