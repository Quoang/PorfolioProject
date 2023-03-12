#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 10:16:52 2021

@author: poniz
"""

import numpy as np
from matplotlib import pyplot as plt 
import scipy.stats as stats
import seaborn as sns



#1) generate 1000 numbers following a standard normal distribution and make the histogram
x=np.random.standard_normal(1000)
plt.hist(x) #numpy array


#2) generate 1 million numbers following a standard normal distribution and make the histogram
y=np.random.standard_normal(1000000)
plt.hist(y)

#3) generate 10 000 numbers following a normal distribution with mean 10 and standard deviation of 4. Make the histogram
z=np.random.normal(10,4,10000)
plt.hist(z)
plt.hist(z,cumulative=True, bins=30)


#4) generate 1000 numbers following an exponential distribution with parameter 4. Make the histogram
a=np.random.exponential(4,1000)
plt.hist(a)
plt.hist(a,cumulative=True, bins=30)


#5) Generate 1 million numbers following a poisson distribution with parameter 4. Make the histogram
b=np.random.poisson(4,1000000)
plt.hist(b)
plt.hist(b,cumulative=True, bins=30)


#6) Generate 1000 numbers following a logistic distribution with parameters 0 and 1. Make the histogram
c=np.random.logistic(0,1,1000)
plt.hist(c)
plt.hist(c,cumulative=True, bins=30)

#7) What is the probability that a random variable X that follows a normal distribution N(15,4) takes values lower than 0?
stats.norm.cdf(0,15,4)

 
#8) What is the probability that a random variable X that follows a standard normal distribution takes values lower than 0?
stats.norm.cdf(0,0,1)

#9) What is the probability that a random variable X that follows a standard normal distribution takes values over 1.96?
stats.norm.sf(1.96)

#Kernel density function
x=np.random.standard_normal(10000)
y=np.random.logistic(0,1,10000)
sns.kdeplot(x, cumulative=True, label="normal")
sns.kdeplot(y, cumulative=True, label="logistic")
plt.show()

sns.kdeplot(x, label="normal")
sns.kdeplot(y, label="logistic")
plt.show()
print(stats.skew(x))
print(stats.skew(y))

print(stats.kurtosis(x))
print(stats.kurtosis(y))




