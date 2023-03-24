import argparse
import os
import subprocess

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--input-folder", type=str, help='Path to folder that contains video files')
    parser.add_argument("-fps", type=float, help='Target FPS', default=25.0)
    parser.add_argument("-o", "--output-folder", type=str, help='Path to output folder')
    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)

    fileList = []
    for root, dirnames, filenames in os.walk(args.input_folder):
        for filename in filenames:
            if os.path.splitext(filename)[1] == '.mp4' or os.path.splitext(filename)[1] == '.mpg' or os.path.splitext(filename)[1] == '.mov' or os.path.splitext(filename)[1] == '.flv':
                fileList.append(os.path.join(root, filename))

    for file in fileList:
        subprocess.run("ffmpeg -i {} -r 25 -y {}".format(file, os.path.splitext(file.replace(args.input_folder, args.output_folder))[0]+".mp4"), shell=True)