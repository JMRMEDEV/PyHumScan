import numpy as np
from PIL import Image

def main():
    color_array = np.array(
        [
        [[252, 3, 152, 255],[255, 255, 255, 255],[255, 255, 255, 255],[255, 255, 255, 255]],
        [[255, 255, 255, 255],[255, 255, 255, 255],[255, 255, 255, 255],[255, 255, 255, 255]],
        [[255, 255, 255, 255],[252, 3, 152, 255],[255, 255, 255, 255],[255, 255, 255, 255]],
        [[255, 255, 255, 255],[255, 255, 255, 255],[255, 255, 255, 255],[252, 3, 152, 255]],
        [[255, 255, 255, 255],[255, 255, 255, 255],[255, 255, 255, 255],[252, 3, 152, 255]]
        ], dtype=np.uint8)
    print(color_array.shape[0])
    
    for x in range(color_array.shape[0]):
        for y in range(color_array.shape[1]):
            if np.array_equal(color_array[x][y], [252, 3, 152, 255]):
                print("Pink detected in ", x, y, "\n\n")

if __name__ == "__main__":
    main()
