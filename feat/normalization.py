import numpy as n

__all__ = [
    'normalize'
]


####################################################################
# - Switch over 5 types of normalization :
def normalize(A, B, norm=0):
    if norm == 0:
        return A
    elif norm == 1:
        return A - B  # 1 = Substraction
    elif norm == 2:
        return A / B  # 2 = Division
    elif norm == 3:
        return (A - B) / B  # 3 - Substract then divide
    elif norm == 4:
        return (A - B) / n.std(B, axis=0)  # 4 - Z-score
