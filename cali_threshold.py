import argparse
import cv2
import numpy as np
import sys, os, copy, math
import logging
from ctypes import *


def is_all_zero(data):
    for num in data:
        if num != 0:
            return False
    return True

# customer may have network output all zero, change it to 1e-5 for them.
def warn_zeros(layer_name, t):
    print("WARNING: layer {} is all zeros. Please check the input data "
          "correctness.".format(layer_name))
    print("WARNING: Set zero value to 1e-5.")
    t[0] = 1e-5

class KLD_Calibrator(object):
    def __init__(self, data,
            input_num=1, histogram_bin_num=2048, auto_tune=True, tune_iteration=10
            , math_lib_path="calibration_math.so"):

        self.data = data
        self.input_num = input_num
        self.histogram_bin_num = histogram_bin_num
        self.calibration_math = CDLL(math_lib_path)
        self.calibration_math.kl_diversity.restype = c_float
        self.calibration_math.kl_diversity_hist.restype = c_float

    def do_find_min_max(self):
        data_max = {}
        data_min = {}
        idx = 0
    
        for item in self.data.files:
          if item not in data_max:
              data_max[item] = 0
              data_min[item] = 0
    
          t = np.abs(self.data[item].flatten())
    
          if t.size > 0:
              if is_all_zero(t):
                  warn_zeros(item, t)
              data_max[item] = max(data_max[item], np.max(t))
              data_min[item] = min(data_min[item], np.min(self.data[item].flatten()))
        return data_min, data_max
    
    def do_histogram(self, data_max):
        data_hist = {}
        width_hist = {}
        idx = 0
        for item in self.data.files:
            t = np.abs(self.data[item].flatten())
            t = t[t!=0]
    
            width = data_max[item] / (self.histogram_bin_num - 1)
            if t.size > 0:
                hist, bins = np.histogram(np.floor(t / width + 0.5),
                                          bins=self.histogram_bin_num,
                                          range=(0, self.histogram_bin_num-1),
                                          density=False)
            else:
                hist = np.zeros(self.histogram_bin_num)
            hist = hist.astype(np.int32)
    
            if item not in data_hist:
                data_hist[item] = hist
                width_hist[item] = width
            else:
                data_hist[item] += hist
    
        return data_hist, width_hist
    
    def KLD_hist(self, data_hist, width):
        return self.calibration_math.kl_diversity_hist(
            data_hist.ctypes.data_as(POINTER(c_int)), c_float(width),
            c_longlong(self.histogram_bin_num))
    
    def do_calibration(self, threshold_table=None):
         data_min, data_max = self.do_find_min_max()
         data_hist, width_hist = self.do_histogram(data_max)
    
         thresholds = {}
         for item in data_hist:
             thresholds[item] = [self.KLD_hist(data_hist[item], width_hist[item])]
    
         if threshold_table:
             with open(threshold_table, 'w') as outfile:
                  for key in thresholds.keys():
                      line = key + ' ' + str(thresholds[key])
                      outfile.write(line)
                      outfile.write('\n')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--blobs_file', metavar='blobs_file', help='blobs file')
    parser.add_argument('--output_file', metavar='output_file', help='output file')
    args = parser.parse_args()
    data = np.load(args.blobs_file)
    calibrator = KLD_Calibrator(data)
    calibrator.do_calibration(args.output_file)

if __name__ == '__main__':
    main()
