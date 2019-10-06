import numpy as np
from scipy.special import hankel2

def viscoelastic_2D(vp, vs, rho, taup, taus, omega0, dt, rec_pos, src):
    """
    Analytic solution for a point force in the z direction in an infinite
    homogeneous space

     vp: P-wave velocity
     vs: S-wave velocity
     rho: density
     taup: relaxation time for P-waves
     taus: relaxation time for S-waves
     omega0: List of relaxation frequencies
     dt: time step size
     rec_pos: a list of [ [x,y,z] ] for each receiver position
     src: the src signal

     The analytic solution can be found in:
     Gosselin-Cliche, B., & Giroux, B. (2014).
     3D frequency-domain finite-difference viscoelastic-wave modeling
     using weighted average 27-point operators with optimal coefficients.
     Geophysics, 79(3), T169-T188. doi: 10.1190/geo2013-0368.1
    """

    nt = src.shape[0]
    nrec = len(rec_pos)
    F = np.fft.fft(src)
    omega = 2*np.pi*np.fft.fftfreq(F.shape[0], dt)

    Vx = np.zeros([nt, nrec], dtype=np.complex128)
    Vz = np.zeros([nt, nrec], dtype=np.complex128)

    for ii in range(1, nt//2):
        # Complex modulus given by the standard linear solid
        fact1 = 0
        fact2 = 0
        for l in range(len(omega0)):
            den = 1.0 + omega[ii] ** 2 / omega0[l] ** 2
            fact1 += omega[ii] ** 2 / omega0[l] ** 2 / den
            fact2 += omega[ii] / omega0[l] / den

        if taup > 0:
            qp = (1.0 + fact1 * taup) / (fact2 * taup)
        else:
            qp = np.inf
        if taus > 0:
            qs = (1.0 + fact1 * taus) / (fact2 * taus)
        else:
            qs = np.inf

        nu = vp ** 2 * rho * (1.0 + 1j / qp) / (1.0 + fact1 * taup)
        mu = vs ** 2 * rho * (1.0 + 1j / qs) / (1.0 + fact1 * taus)

        kp = omega[ii] / np.sqrt(nu / rho)
        ks = omega[ii] / np.sqrt(mu / rho)
        vp = np.sqrt(nu/rho)
        vs = np.sqrt(mu/rho)

        for jj in range(nrec):
            x, _, z = rec_pos[jj]

            r = np.sqrt(x ** 2 + z ** 2)

            G1 = -1j * np.pi / 2 * (1 / vp ** 2 * hankel2(0, kp * r) +
                      1. / (omega[ii] * r * vs) * hankel2(1, ks * r) -
                      1. / (omega[ii] * r * vp) * hankel2(1, kp * r))
            G2 =  1j * np.pi / 2 * (1 / vs ** 2 * hankel2(0, ks * r ) -
                      1. / (omega[ii] * r * vs) * hankel2(1, ks * r ) +
                      1. / (omega[ii] * r * vp) * hankel2(1, kp * r ))
            print(hankel2(1, ks * r),  ks * r)
            Vx[ii, jj] = F[ii] / (2 * np.pi * rho) * (x * z / r ** 2) * (G1 + G2) * 1j * omega[ii]
            Vz[ii, jj] = F[ii] / (2 * np.pi * rho) * (1 / r ** 2) * (z ** 2 * G1 - x ** 2 * G2) * 1j * omega[ii]

    for ii in range(1, nt//2):
        Vx[-ii, :] = np.conj(Vx[ii, :])
        Vz[-ii, :] = np.conj(Vz[ii, :])

    vx = nt * np.fft.ifft(Vx, axis=0)
    vz = nt * np.fft.ifft(Vz, axis=0)
    vx = np.abs(vx) * np.sign(np.real(vx))
    vz = np.abs(vz) * np.sign(np.real(vz))

    return vx, vz
