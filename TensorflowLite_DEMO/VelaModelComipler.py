import os
import argparse

def vela_compiler_tool(PATH):
    for dirPath, dirNames, fileNames in os.walk(PATH):
        for file in fileNames:
            filename, extension = os.path.splitext(file)
            if extension == ".tflite" :
                if filename[-4:] != "vela" : # ignore vela model
                    cmd = 'vela {}'.format(file);print(cmd)
                    os.chdir(dirPath)
                    os.system(cmd)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p',"--path", default="/home/root/TensorflowLite_DEMO/_experiment_")
    args = parser.parse_args()
    vela_compiler_tool(args.path)

if __name__ == "__main__":
    main()