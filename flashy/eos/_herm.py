import numpy as np


# quintic hermite polynomial statement functions
# psi0 and its derivatives
psi0   = lambda zFunc: zFunc**3 * (zFunc * (-6.0*zFunc + 15.0) - 10.0) + 1.0
dpsi0  = lambda zFunc: zFunc**2 * (zFunc * (-30.0*zFunc + 60.0) - 30.0)
ddpsi0 = lambda zFunc: zFunc * (zFunc * (-120.0*zFunc + 180.0) - 60.0)

# psi1 and its derivatives
psi1   = lambda zFunc: zFunc * (zFunc**2 * (zFunc * (-3.0*zFunc + 8.0) - 6.0) + 1.0)
dpsi1  = lambda zFunc: zFunc**2 * (zFunc * (-15.0*zFunc + 32.0) - 18.0) + 1.0
ddpsi1 = lambda zFunc: zFunc * (zFunc * (-60.0*zFunc + 96.0) - 36.0)

# psi2 and its derivatives
psi2   = lambda zFunc: 0.5 * zFunc**2 * (zFunc * (zFunc * (-zFunc + 3.0) - 3.0) + 1.0)
dpsi2  = lambda zFunc: 0.5 * zFunc * (zFunc * (zFunc * (-5.0*zFunc + 12.0) - 9.0) + 2.0)
ddpsi2 = lambda zFunc: 0.5 * (zFunc * (zFunc * (-20.0*zFunc + 36.0) - 18.0) + 2.0)

# cubic hermite polynomial statement functions
#  psi0, derivative unused
xpsi0 = lambda zFunc: zFunc * zFunc * (2.0*zFunc - 3.0) + 1.0
#  psi1, derivative unused
xpsi1 = lambda zFunc: zFunc * (zFunc * (zFunc - 2.0) + 1.0)


# # Biquintic hermite polynomial for the free energy
def herm5(w0t, w1t, w2t, w0mt, w1mt, w2mt, w0d, w1d, w2d, w0md, w1md, w2md, fi):
    weights = np.array([
        w0d * w0t,   w0md * w0t,   w0d * w0mt,   w0md * w0mt,
        w0d * w1t,   w0md * w1t,   w0d * w1mt,   w0md * w1mt,
        w0d * w2t,   w0md * w2t,   w0d * w2mt,   w0md * w2mt,
        w1d * w0t,   w1md * w0t,   w1d * w0mt,   w1md * w0mt,
        w2d * w0t,   w2md * w0t,   w2d * w0mt,   w2md * w0mt,
        w1d * w1t,   w1md * w1t,   w1d * w1mt,   w1md * w1mt,
        w2d * w1t,   w2md * w1t,   w2d * w1mt,   w2md * w1mt,
        w1d * w2t,   w1md * w2t,   w1d * w2mt,   w1md * w2mt,
        w2d * w2t,   w2md * w2t,   w2d * w2mt,   w2md * w2mt
    ])

    return np.dot(fi, weights)


# Bicubic hermite polynomial for the electron pressure derivatives
def herm3dpd(w0t, w1t, w0mt, w1mt, w0d, w1d, w0md, w1md, fi):
    weights = np.array([
        w0d * w0t,   w0md * w0t,   w0d * w0mt,   w0md * w0mt,
        w0d * w1t,   w0md * w1t,   w0d * w1mt,   w0md * w1mt,
        w1d * w0t,   w1md * w0t,   w1d * w0mt,   w1md * w0mt,
        w1d * w1t,   w1md * w1t,   w1d * w1mt,   w1md * w1mt
    ])

    return np.dot(fi, weights)


# Bicubic hermite polynomial for the electron chemical potential
def herm3e(w0t, w1t, w0mt, w1mt, w0d, w1d, w0md, w1md, fi):
    weights = np.array([
        w0d * w0t,   w0md * w0t,   w0d * w0mt,   w0md * w0mt,
        w0d * w1t,   w0md * w1t,   w0d * w1mt,   w0md * w1mt,
        w1d * w0t,   w1md * w0t,   w1d * w0mt,   w1md * w0mt,
        w1d * w1t,   w1md * w1t,   w1d * w1mt,   w1md * w1mt
    ])

    return np.dot(fi, weights)

# Bicubic hermite polynomial for the electron positron number densities
def herm3x(w0t, w1t, w0mt, w1mt, w0d, w1d, w0md, w1md, fi):
    weights = np.array([
        w0d * w0t,   w0md * w0t,   w0d * w0mt,   w0md * w0mt,
        w0d * w1t,   w0md * w1t,   w0d * w1mt,   w0md * w1mt,
        w1d * w0t,   w1md * w0t,   w1d * w0mt,   w1md * w0mt,
        w1d * w1t,   w1md * w1t,   w1d * w1mt,   w1md * w1mt
    ])

    return np.dot(fi, weights)
