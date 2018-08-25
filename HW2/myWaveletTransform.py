import numpy as np
import pywt


def forward_transform(y, wavelet):
    [cA3, (cH3, cV3, cD3), (cH2, cV2, cD2), (cH1, cV1, cD1)] = pywt.wavedec2(y, wavelet, 'per', 3)
    cA2 = np.vstack((np.hstack((cA3, cH3)), np.hstack((cV3, cD3))))
    cA1 = np.vstack((np.hstack((cA2, cH2)), np.hstack((cV2, cD2))))
    return np.vstack((np.hstack((cA1, cH1)), np.hstack((cV1, cD1))))


def inverse_transform(x, wavelet):
    cA1 = np.vsplit(np.hsplit(x, 2)[0], 2)[0]
    cV1 = np.vsplit(np.hsplit(x, 2)[0], 2)[1]
    cH1 = np.vsplit(np.hsplit(x, 2)[1], 2)[0]
    cD1 = np.vsplit(np.hsplit(x, 2)[1], 2)[1]
    cA2 = np.vsplit(np.hsplit(cA1, 2)[0], 2)[0]
    cV2 = np.vsplit(np.hsplit(cA1, 2)[0], 2)[1]
    cH2 = np.vsplit(np.hsplit(cA1, 2)[1], 2)[0]
    cD2 = np.vsplit(np.hsplit(cA1, 2)[1], 2)[1]
    cA3 = np.vsplit(np.hsplit(cA2, 2)[0], 2)[0]
    cV3 = np.vsplit(np.hsplit(cA2, 2)[0], 2)[1]
    cH3 = np.vsplit(np.hsplit(cA2, 2)[1], 2)[0]
    cD3 = np.vsplit(np.hsplit(cA2, 2)[1], 2)[1]
    return pywt.waverec2([cA3, (cH3, cV3, cD3), (cH2, cV2, cD2), (cH1, cV1, cD1)], wavelet, 'per')