import numpy as np
from autrainer.models.cnn_10 import Cnn10
from autrainer.models.cnn_14 import Cnn14
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import (
    show_cam_on_image, show_cam_on_image
)
import cv2, os, torch
from PIL import Image


def main(model_path, save_path, test_path):


    device = ("cuda" if torch.cuda.is_available() else "cpu")
    model = Cnn10(output_dim=2, in_channels=1, segmentwise=False)
    
    checkpoint = torch.load(model_path, map_location="cpu")
    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    model.to(device)
    test_location = test_path
    save_location = save_path
    os.makedirs(save_location, exist_ok=True)

    for spectrogram_name in os.listdir(test_location):
        determinant = spectrogram_name.split("_")[0]
        if "CM" in determinant or "CF" in determinant or "MC" in determinant or "FC" in determinant:
            classifier = 0
        
        elif "M" in determinant or "F" in determinant:
            classifier = 1


        spectrogram = np.load(os.path.join(test_location, spectrogram_name))
        spectrogram = np.squeeze(spectrogram)
        input_tensor = torch.tensor(spectrogram, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

        target_layers = [model.conv_block4.conv2]
        cam = GradCAM(model=model, target_layers=target_layers)
        targets = [ClassifierOutputTarget(int(classifier))]  
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)


        grayscale_cam = grayscale_cam[0, :]
        
        nor = (spectrogram - spectrogram.min()) / (spectrogram.max() - spectrogram.min())
        nor = nor.astype(np.float32)
        

        spec_rgb = np.stack([nor, nor, nor], axis=-1)
        
        spec_rgb = np.transpose(spec_rgb, (1, 0, 2))
        
        
        grayscale_cam = np.transpose(grayscale_cam, (1, 0))
        
        spec_rgb_botleft = np.flipud(spec_rgb)
        grayscale_cam_botleft = np.flipud(grayscale_cam)

        cam_image = show_cam_on_image(spec_rgb_botleft, grayscale_cam_botleft, use_rgb=True)
        
        folder_name = spectrogram_name.split(".")[0]
        os.makedirs(save_location, exist_ok=True)
        os.makedirs(os.path.join(save_location, folder_name), exist_ok=True)
        np.save(f"{save_location}/{folder_name}/gradcam.npy", grayscale_cam_botleft)
        cv2.imwrite(f"{save_location}/{folder_name}/gradcam.jpg", cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR))
        normal_image = (nor * 255).astype(np.uint8)
        normal_image = np.transpose(normal_image, (1, 0))
        normal_image = np.flipud(normal_image)
        img = Image.fromarray(normal_image)
        img.save(f"{save_location}/{folder_name}/normal.jpg")




main()