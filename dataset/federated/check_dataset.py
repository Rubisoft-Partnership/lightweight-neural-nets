import os
import numpy as np

def display(img, width=28, threshold=200):
        render = ''
        for i in range(len(img)):
            if i % width == 0:
                render += '\n'
            if img[i] > threshold:
                render += '@'
            else:
                render += '.'
        return render
    
    
def main():
    path = os.sys.argv[1]
    if not os.path.exists(path):
        print("Invalid path")
        return
    
    images, labels = parse_dataset(path)
    
    # Print 10 random images and their labels
    for i in range(10):
        index = np.random.randint(0, len(images))
        print("Label: ", labels[index])
        width = np.floor(np.sqrt(len(images[index])))
        print(display(images[index], width=width, threshold=0.5))
            
            
def parse_dataset(path):
    images = []
    labels = []
    
    with open(path, 'r') as f:
        # Read space-separated floats
        for line in f:
            line = [float(x) for x in line.split()]
            one_hot_label = line[-10:]
            label = one_hot_label.index(1)
            labels.append(label)
            images.append(line[:-10])
            
    return images, labels
    
    
if __name__ == '__main__':
    main()