# -*- coding: utf-8 -*-
"""
Created on Mon May 23 00:13:11 2022

@author: harsh
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_loss(loss_unstructured, loss_structured):
    ratio = []
    losses_un = []
    losses_st = []
    for i in loss_unstructured:
        r, ls = i[0], i[1]
        ratio.append(float("{:.2f}".format(r)))
        losses_un.append(ls)

    for i in loss_structured:
        r, ls = i[0], i[1]
        losses_st.append(ls)

    fig1 =  plt.figure(figsize=(15, 15))
    plt.clf()
    plt.ylim(0,5)
    plt.subplot(211)
    plt.title("Test loss ")
    plt.plot(losses_st, 'g-')
    plt.plot(losses_un, 'b-')
    default_x_ticks = range(len(ratio))
    #plt.plot(default_x_ticks, y)
    plt.xticks(default_x_ticks, ratio)

    plt.ylabel('Loss')
    plt.xlabel('Sparsity')
    plt.legend(['Structured','UnStructured'], loc='upper right')
    plt.show() 



def plot_loss(accuracy_unstructured, accuracy_structured):
    ratio = []
    acc_uns = []
    acc_str = []

    for i in accuracy_unstructured:
        r, ac = i[0], i[1]
        ratio.append(float("{:.2f}".format(r)))
        acc_uns.append(ac)

    for i in accuracy_structured:
        r, ac = i[0], i[1]
        acc_str.append(ac)

    fig1 =  plt.figure(figsize=(15, 15))
    plt.clf()
    plt.ylim(0,5)
    plt.subplot(211)
    plt.title("Test Accuray ")
    plt.plot(acc_str, 'b-')
    plt.plot(acc_uns, 'g-')
    default_x_ticks = range(len(ratio))
    #plt.plot(default_x_ticks, y)
    plt.xticks(default_x_ticks, ratio)

    plt.ylabel('Accuracy')
    plt.xlabel('Sparsity')
    plt.legend(['Structured', 'UnStructured'], loc='upper right')
    plt.show() 
