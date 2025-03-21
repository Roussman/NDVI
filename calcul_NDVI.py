

import matplotlib.pyplot as plt
import argparse
from dask.array.image import imread as da_imread
import dask.array as da

def readArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_files",
                        help="Path to input file", type=str)
    parser.add_argument("-o", "--output_folder",
                        help="Path to output figures folder", type=str)
    args = parser.parse_args()
    return args
  
def realizeNDVI(inputImage):
    return (inputImage[:, :, :, 2] - inputImage[:, :, :, 3]) / (inputImage[:, :, :, 3] + inputImage[:, :, :, 2])

def readFileFromPath(path):
    images_array = da_imread(path)
    return images_array

def prepareNDVI(images_array):
    big_images_array = da.concatenate([images_array] * 10)
    big_images_array = big_images_array.rechunk((1, 4050, 4550, 4))
    ndviTIF = realizeNDVI(big_images_array)
    meanNdviTIF = da.nanmean(ndviTIF, axis=(1,2))
    return ndviTIF, meanNdviTIF

def plotFigures(ndviTIF, meanNdviTIF, output_path):
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.imshow(ndviTIF[0].compute())
    plt.savefig("{0}/ndvi.png".format(output_path))
    
    fig, ax = plt.subplots(nrows=1, ncols=1)
    plt.plot(meanNdviTIF.compute())
    plt.savefig("{0}/meanndvi.png".format(output_path))
    return

if __name__ == '__main__':
  args = readArgs()
  ndviTIF, meanNdviTIF = prepareNDVI(readFileFromPath(args.input_files))
  plotFigures(ndviTIF, meanNdviTIF, args.output_folder)
