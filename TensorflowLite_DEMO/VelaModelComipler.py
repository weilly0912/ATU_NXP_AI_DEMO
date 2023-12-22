import os
import argparse
import subprocess

def vela_compiler_tool(PATH):
    for dirPath, dirNames, fileNames in os.walk(PATH):
        for file in fileNames:
            filename, extension = os.path.splitext(file)
            if extension == ".tflite" :
                if filename[-4:] != "vela" : # ignore vela model
                    # gerenate vela model
                    cmd  = 'vela {} --output-dir .'.format(file);print(cmd)
                    os.chdir(dirPath)#os.chdir(os.getcwd() + dirPath[1:])
                    try :
                        output = subprocess.check_output(cmd, shell=True)
                        os.system('rm *.csv')
                    except :
                        pass

                    # save result
                    os.system('mkdir -p detail')
                    with open( "detail/" + file + ".txt", "w") as file:
                        file.write(output.decode("utf-8"))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p',"--path", default="/home/root/TensorflowLite_DEMO")
    args = parser.parse_args()
    vela_compiler_tool(args.path)

if __name__ == "__main__":
    main()