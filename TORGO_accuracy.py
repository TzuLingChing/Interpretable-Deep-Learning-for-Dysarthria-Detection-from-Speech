import numpy as np
from autrainer.models.cnn_10 import Cnn10
from autrainer.models.cnn_14 import Cnn14
import torch
from torchvision.datasets import DatasetFolder
from torch.utils.data import DataLoader




def npyloader_fixed(path):
    spectrogram = np.load(path)
    return torch.from_numpy(spectrogram).float()



def eval_random3s(model_path):

    device = ("cuda" if torch.cuda.is_available() else "cpu")
    #Manuelly change the model according to the model you used
    model = Cnn10(output_dim=2, in_channels=1, segmentwise=False)
    checkpoint = torch.load(model_path, map_location="cpu")
    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    model.to(device)
    y_true = []
    correct = 0
    total_samples = 0
    y_pred = []
    test_path = "torgo"
    test_dataset = DatasetFolder(root = test_path, loader = npyloader_fixed ,extensions= (".npy",))

    test_dataloader = DataLoader(test_dataset, batch_size=32)

    with torch.no_grad():
        for spectrograms, labels in test_dataloader:
            
            spectrograms = spectrograms.to(device)
            labels = labels.to(device)

            outputs = model(spectrograms)

            _, predicted = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy())  
            y_pred.extend(predicted.cpu().numpy())
            total_samples += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total_samples


        print(f"The accuracy of input model has an accuracy of : {accuracy:.2f}%")


def main():
    eval_random3s()

main()