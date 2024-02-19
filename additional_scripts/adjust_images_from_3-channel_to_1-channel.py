import os
from PIL import Image
import argparse

def convert_to_grayscale(input_path, output_path):
    image = Image.open(input_path)
    gray_image = image.convert('L')
    gray_image.save(output_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Provide input parameters")

    parser.add_argument("--pth", dest="path", default="D:\SAN\datasets\coco-active_speaker-active_speaker-overlapped\stuffthingmaps", type=str, action="store")
    # Directory path to start the search
    root_directory = parser.path

    # Loop through all the files and subdirectories
    for root, dirs, files in os.walk(root_directory):
        for file in files:
            # Check if the file is a jpg or png image
            if file.lower().endswith(('.jpg', '.png')):
                input_path = os.path.join(root, file)
                if file.lower().endswith('.jpg'):
                    outputpath = input_path.replace(".jpg", ".png")
                else:
                    outputpath = input_path
                # output_path = os.path.join(root, f'grayscale_{file}')

                # Convert the image to grayscale
                convert_to_grayscale(input_path, outputpath)
                print(f'Converted {file} to grayscale.')

    print('Conversion complete.')