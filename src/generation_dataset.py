import os
import random
import numpy as np
from enum import Enum

if not os.path.exists('data'):
    os.mkdir('data')

class Parameters(Enum):
    DICE_FACE_NUM:int=2
    DICE_A_PROB:list=[0.8, 0.2]
    DICE_B_PROB:list=[0.6, 0.4]
    DICE_C_PROB:list=[0.3, 0.7]
    DICES_PROB:list=[0.1, 0.4, 0.5]
    DATA_LENGTH:int=10000

class Dice:
    def __init__(self, face_num:int, prob:list) -> None:
        self._face_num = face_num
        self._prob = prob
    
    def roll(self) -> int:
        return random.choices(range(1, self._face_num+1), self._prob)[0]

diceA = Dice(Parameters.DICE_FACE_NUM.value, Parameters.DICE_A_PROB.value)
diceB = Dice(Parameters.DICE_FACE_NUM.value, Parameters.DICE_B_PROB.value)
diceC = Dice(Parameters.DICE_FACE_NUM.value, Parameters.DICE_C_PROB.value)

diceList = [diceA, diceB, diceC]

datas = []

for n in range(Parameters.DATA_LENGTH.value):
    dice = random.choices(diceList, Parameters.DICES_PROB.value)[0]    
    datas.append(dice.roll())

np.savetxt('./data/dataset.txt', datas, fmt='%d')
print("Data is generation!!")
