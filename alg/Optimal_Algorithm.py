import math
from itertools import combinations

from env.evaluation import evaluation
from env.large_scale_evaluation import large_scale_evalutaion


def opt_alg(M1, M2, M3, N1, N2, N3, A, x, eta):
    mn = math.inf
    res = []
    for s1 in combinations(range(N1), M1):
        for s2 in combinations(range(N1, N1 + N2), M2):
            for s3 in combinations(range(N1 + N2, N1 + N2 + N3), M3):
                s = list(s1 + s2 + s3)
                tmp = evaluation(A, x, eta, s)
                if tmp < mn:
                    mn = tmp
                    res = s
    return res

def opt_alg2(M1, M2, N1, N2, A, x, eta):
    mn = math.inf
    res = []
    for s1 in combinations(range(N1), M1):
        for s2 in combinations(range(N1, N1 + N2), M2):
            s = list(s1 + s2)
            tmp = large_scale_evalutaion(A, x, eta, s)
            if tmp < mn:
                mn = tmp
                res = s
    return res