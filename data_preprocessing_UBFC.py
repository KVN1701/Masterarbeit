import matplotlib.pyplot as plt
import numpy as np
from pyts.image import MarkovTransitionField
import os
from tqdm import tqdm

# ! Daten sind 180 Sekunden lang

mtf = MarkovTransitionField(n_bins=8, strategy='uniform')


def plot_markov(data_arr):
    X_mtf = mtf.fit_transform(data_arr)
    
    fig = plt.figure(figsize=(5, 5))
    
    ax_mtf = fig.add_subplot()
    ax_mtf.imshow(X_mtf[0], cmap='rainbow', origin='lower', vmin=0., vmax=1.)
    
    return plt
    

def generate_dataset_timespans(length: int=30, delay: int=5):
    ret = []
    file_path = 'UBFC-Phys'
    
    for subdir in os.listdir(f'{file_path}/'):
        for file_name in os.listdir(f'{file_path}/{subdir}/'):
            # handling bvp files
            if 'bvp' in file_name:
                continue # ! Vorübergehend nur EDA-Daten
                hz = 64 # bvp-data is in 64 Hz
            
            # handling eda files
            elif 'eda' in file_name:
                hz = 4 # eda-data is in 4 Hz
                
            # exclude any aditional files
            else:
                continue 
            
            # getting all values within the data file
            values = [float(val.strip()) for val in open(f'{file_path}/{subdir}/{file_name}')]
            value_snippets = []    
            
            # split the values into equal parts
            for start in range(0, len(values) - ((length - delay) * hz), delay * hz):
                stop = start + length * hz
                value_snippets.append(np.array([values[start:stop]]))
            
            ret.extend(value_snippets)
    return ret


if __name__ == '__main__':
    total_data = generate_dataset_timespans(length=15, delay=5)
    
    for i in tqdm(range(len(total_data))):
        arr = total_data[i]
        plot = plot_markov(arr)
        os.makedirs('datasets/ubfc', exist_ok=True)
        plot.savefig(f'datasets/ubfc/plot_{i}.png')
        plt.close()
            