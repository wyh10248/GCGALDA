# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 17:37:48 2023

@author: 28473
"""

import scipy.stats as stats
import numpy as np 

# 两组方法在405种疾病上的AUC分数
# auc1 = [0.9681, 0.94] 
# auc2 = [0.4, 0]

# # 进行Wilcoxon签名秩检验
# z_value, p_value = stats.wilcoxon(x=auc1, y=auc2)

# print("z统计量:", z_value)
# print("p值:", p_value)

# # 输出结果
# if p_value < 0.05:
#     print("两组AUC差异具有统计学显著性")
# else:
#     print("两组AUC差异不具统计学显著性")
import scipy.stats as stats
import numpy as np 

# 生成两组具有显著性差异的数据
np.random.seed(0)
# 第一组数据从正态分布生成
data1 = np.random.normal(loc=0, scale=1, size=100)
# 第二组数据从平移后的正态分布生成
data2 = np.random.normal(loc=0.5, scale=1, size=100)

# 进行Wilcoxon签名秩检验
z_value, p_value = stats.wilcoxon(x=data1, y=data2)

print("z统计量:", z_value)
print("p值:", p_value)

# 输出结果
if p_value < 0.05:
    print("两组数据差异具有统计学显著性")
else:
    print("两组数据差异不具统计学显著性")

