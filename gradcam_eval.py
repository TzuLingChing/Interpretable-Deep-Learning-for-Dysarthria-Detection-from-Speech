import numpy as np
import os
import matplotlib.pyplot as plt

def calc_average_gradcam(gradcam_path):

    file_location = gradcam_path


    average_gradcam_map_control = np.zeros((64,301), dtype = float)
    average_gradcam_map_non = np.zeros((64,301), dtype = float)
    total_amount_control = 0.0
    total_amount_non = 0.0
    for speakers in os.listdir(file_location):
        if os.path.isdir(os.path.join(file_location, speakers)):
            for spec in os.listdir(os.path.join(file_location, speakers)):
                if speakers.startswith("C"):
                    if spec.endswith(".npy"):
                        gradcam = np.load(os.path.join(file_location, speakers, spec))
                        mx = gradcam.max()
                        if mx > 0:
                            gradcam = gradcam / mx
                        if gradcam.shape == (64, 301):
                            total_amount_control += 1.0
                            average_gradcam_map_control = np.add(average_gradcam_map_control, gradcam)
                else:
                    if spec.endswith(".npy"):
                        gradcam = np.load(os.path.join(file_location, speakers, spec))
                        mx = gradcam.max()
                        if mx > 0:
                            gradcam = gradcam / mx
                        if gradcam.shape == (64, 301):
                            total_amount_non += 1.0
                            average_gradcam_map_non = np.add(average_gradcam_map_non, gradcam)
    if total_amount_control > 0.0: 
        average_gradcam_map_control = np.divide(average_gradcam_map_control, total_amount_control)
        np.save(f"{gradcam_path}/average_control_group_gradcam.npy", average_gradcam_map_control)  
    
    if total_amount_non > 0.0: 
        average_gradcam_map_non = np.divide(average_gradcam_map_non, total_amount_non)
        np.save(f"{gradcam_path}/average_dysarthria_group_gradcam.npy", average_gradcam_map_non)  
    
    
def compute_class_maps(gradcam_path):
    
    classess = ["average_control_group_gradcam", "average_dysarthria_group_gradcam"]
    
    for classes in classess:
        avg_map = np.load(f"{gradcam_path}/{classes}.npy")
        avg_map = np.flipud(avg_map)
        height, width =  avg_map.shape
        blank = np.ones((height, width, 3), dtype=float)

        ax = plt.subplots(figsize=(12, 4))
        ax.imshow(blank, origin="lower", aspect="auto")

        hm = ax.imshow(avg_map, origin="lower", cmap="magma" ,aspect="auto")

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Mel bin")
        ax.set_xticks([0, width//3, 2*width//3, width-1])
        ax.set_xticklabels(["0", "1", "2", "3"])    

        plt.colorbar(hm, ax)
        plt.savefig(f"{gradcam_path}/{classes}.png", dpi=100)



def main():
    gradcam_path = ""
    #calc_average_gradcam(gradcam_path)
    #compute_class_maps(gradcam_path)
    

    

main()


