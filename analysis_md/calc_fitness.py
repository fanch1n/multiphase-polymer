import numpy as np

from scipy.optimize import curve_fit
from scipy.optimize import differential_evolution
from scipy.interpolate import CubicSpline
from scipy.integrate import quad
from scipy.optimize import minimize

import warnings
import json
import os
from glob import glob

import matplotlib.pyplot as plt


# function for genetic algorithm to minimize (sum of squared error)
def sumOfSquaredError(parameterTuple, xData, yData, func):
    warnings.filterwarnings("ignore")  # do not print warnings by genetic algorithm
    val = func(xData, *parameterTuple)
    return np.sum((yData - val) ** 2.0)


def unitstep(x, start, end, scale=0.5):
    return scale * (1 + np.sign(x - start)) - scale * (1 + np.sign(x - end))


def right_step(x, a, b, scale=0.5):
    return scale * unitstep(x, a, b) + unitstep(x, b, b + (b - a))


def left_step(x, a, b, scale=0.5):
    return unitstep(x, a, b) + scale * unitstep(x, b, b + (b - a))


def generate_Initial_Parameters(xData, yData, func):
    # min and max used for bounds
    maxX = max(xData)
    minX = min(xData)
    parameterBounds = []
    parameterBounds.append([minX, maxX])  # search bounds for a
    parameterBounds.append([minX, maxX])  # search bounds for b
    # "seed" the numpy random number generator for repeatable results
    result = differential_evolution(
        sumOfSquaredError, parameterBounds, args=(xData, yData, func)
    )
    return result.x


def try_fit(xData, yData, func):
    min_err = float("inf")
    para = None
    rsq = -1
    for i in range(10):
        geneticParameters = generate_Initial_Parameters(xData, yData, func)
        fittedParameters, pcov = curve_fit(func, xData, yData, geneticParameters)
        modelPredictions = func(xData, *fittedParameters)
        absError = modelPredictions - yData
        SE = np.square(absError)  # squared errors
        MSE = np.mean(SE)  # mean squared errors
        RMSE = np.sqrt(MSE)  # Root Mean Squared Error, RMSE
        Rsquared = 1.0 - (np.var(absError) / np.var(yData))
        if RMSE < min_err:
            min_err = RMSE
            para = fittedParameters
            rsq = Rsquared
        # print()
        # print('RMSE:', RMSE)
        # print('R-squared:', Rsquared)
        # print()
    return para, rsq, min_err


# bins = np.linspace(0, 120, 40)
# allpaths = glob("/Users/fanc/della-home/GS/polymer-test/03_04_pipeline_N3/N4-index4/comparison/mineig0.75/*/", recursive = False)

# data = read_atom_file(os.path.join(coexdir, 'final-stitch-%s%s.atom' %(a, b)))
# dat_mol_seq_map = map_Mol_Sequence(np.array(data), dat_phase)

# dat_binned, rhos, compo_dat, dom = bin_data(data, dat_phase, a, b, dat_mol_seq_map, 432, bins, Lx=20)
# x = bins[:-1]
# y = dat_binned[:, 1]
# y = y * density_filter(rhos, 0.1)

# alias data to match pervious example
# phi_alpha = ref_compositions(dat_phase['phases'][str(a)], dat_phase)
# phi_beta = ref_compositions(dat_phase['phases'][str(b)], dat_phase)

# overlap = np.dot(phi_alpha, phi_beta)/np.linalg.norm(phi_alpha, ord=2)/np.linalg.norm(phi_beta, ord=2)

# xData = x
# yData = y
#
# func = lambda x, start, end: test_func(x, start, end, scale=overlap)
#
# fittedParameters, Rsquared, RMSE = try_fit(x, y, func)
# modelPredictions = func(xData, *fittedParameters)
# absError = modelPredictions - yData
# SE = np.square(absError) # squared errors
# MSE = np.mean(SE) # mean squared errors
# RMSE = np.sqrt(MSE) # Root Mean Squared Error, RMSE
# Rsquared = 1.0 - (np.var(absError) / np.var(yData))
# print('Fitted parameters:', fittedParameters)
# print('RMSE:', RMSE)
# print('R-squared:', Rsquared)
# chars = tuple(datapath.split('/')[-2].split('-'))
# if chars not in map_:
#    map_[chars] = []
# map_[chars].append(RMSE)
