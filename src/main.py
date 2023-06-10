# %%
import os
import numpy as np
from enum import Enum
import matplotlib.pyplot as plt

if not os.path.exists('./logs'):
    os.mkdir('./logs')

class Parameters(Enum):
    ROLL_NUM:int=2
    DICE_NUM:int=3
    EPSILON:float=1e-7
    FILE_LLK_NEMA:str="llk.txt"
    FILE_PI_NAME:str="pi.txt"

"""教師なし学習 EMアルゴリズム"""
class Unsupervised_Learning_Algorithm():
    """コンストラクタ"""
    def __init__(self, data_path:str='./dataset/data.txt') -> None:
        self.DATA_PATH = data_path
        self.DATA = np.loadtxt(data_path)
        self.DATA_LENGTH = len(self.DATA)
        # 出目の出現回数
        self.data_hist = np.zeros(Parameters.ROLL_NUM.value)
        # 確率分布P(w|v)
        self._prob = np.zeros((Parameters.DICE_NUM.value, Parameters.ROLL_NUM.value))
        # パラメータπの初期値
        self._pi = np.array([0.3, 0.5, 0.2])
        # パラメータθの初期値(真の分布)
        self._theta = np.array([
                                    [0.8, 0.2], 
                                    [0.6, 0.4], 
                                    [0.3, 0.7]
                                ])
        # 繰り返し計算回数
        self._loop_count = 50
        # 対数尤度
        self._llk = 0

        # パラメータπ 履歴
        self._pi_hist = [self._pi]
        # 対数尤度 履歴
        self._llk_hist = []

    """データ出現回数を計算"""
    def calc_hist(self):
        for n in range(self.DATA_LENGTH):
            self.data_hist[int(self.DATA[n]-1)] += 1
        print(f"hist:{self.data_hist}")
    
    """EMアルゴリズムになる前のやつ"""
    def em_algorithm(self):
        # 事前処理
        # 頻度を計算
        self.calc_hist()

        # step2. ベイズ定理より、確率分布p(w,vを計算)
        self._prob = self._pi * self._theta.T
        self._prob = np.divide( self._prob.T, np.sum(self._prob, axis=1) )
        self._prob = self._prob.T

        # step3. パラメータを再計算
        for cnt in range(self._loop_count):
            # step3-1. パラメータπを計算
            self._pi = self.data_hist * self._prob.T
            self._pi = np.sum(self._pi, axis=1)
            self._pi /= self.DATA_LENGTH

            # パラメータπ 記録
            self._pi_hist.append(self._pi)
            
            # step3-2. パラメータθを計算
            self._prob = self._pi * self._theta.T
            self._prob = np.divide( self._prob.T, np.sum(self._prob, axis=1) )
            self._prob = self._prob.T

            # パラメータθは実験条件により固定
            # self._theta = self.data_hist * self._prob.T
            # self._theta /= np.sum(self._theta)
        
            # step4. 対数尤度を計算 P(x) = #P(vk)^rk
            self._llk = np.sum( self.data_hist * np.log(np.sum( self._pi * self._theta.T, axis=1 ) ) )

            # 対数尤度 記録
            self._llk_hist.append(self._llk)

            print(f"ステップ:{cnt+1}, パラメータπ:{self._pi}, パラメータθ:{self._theta}, 対数尤度:{self._llk}")
        
        np.savetxt('./logs/' + Parameters.FILE_LLK_NEMA.value, self._llk_hist)
        np.savetxt('./logs/' + Parameters.FILE_PI_NAME.value, self._pi_hist)

if __name__ == '__main__':
    ai = Unsupervised_Learning_Algorithm()
    ai.em_algorithm()
# %%
