import gym
import gymnasium
import numpy
import torch
import tianshou
import pygame
import os

packages = [gym, gymnasium, numpy, torch, tianshou, pygame]
pack_name = ["gym", "gymnasium", "numpy", "torch", "tianshou", "pygame"]

for name, p in zip(pack_name, packages):
    print(name, p.__version__)
