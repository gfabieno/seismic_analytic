import numpy as np

def viscoelastic_3D(vp, vs, rho, taup, taus, omega0, dt, rec_pos, src):
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
    Vy = np.zeros([nt, nrec], dtype=np.complex128)
    Vz = np.zeros([nt, nrec], dtype=np.complex128)

    y = 0
    for ii in range(1, nt):
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

        for jj in range(nrec):
            x, y, z = rec_pos[jj]

            R = np.sqrt(x ** 2 + y ** 2 + z ** 2)
            amp = F[ii] / (4.0 * np.pi * rho * R ** 5 * omega[ii] ** 2)

            Vx[ii, jj] = amp * x * z * (
                                        (R ** 2 * kp ** 2
                                         - 3.0 - 3.0 * 1j * R * kp)
                                        * np.exp(-1j * kp * R)

                                        + (3.0 + 3.0 * 1j * R * ks
                                           - R ** 2 * ks ** 2)
                                        * np.exp(-1j * ks * R)
                                        ) * 1j * omega[ii]


            Vy[ii, jj] = amp * y * z * (
                                        (R ** 2 * kp ** 2
                                         - 3 - 3 * 1j * R * kp)
                                        * np.exp(-1j * kp * R) +

                                        (3 + 3 * 1j * R * ks
                                         - R ** 2 * ks ** 2)
                                        * np.exp(-1j * ks * R)
                                        ) * 1j * omega[ii]

            Vz[ii, jj] = amp * (
                                (x ** 2 + y ** 2 - 2.0 * z ** 2)
                                 * (np.exp(-1j * kp * R) - np.exp(-1j * ks * R))

                                + (z ** 2 * R ** 2 * kp ** 2
                                    + 1j * (x ** 2 + y ** 2 - 2.0 * z ** 2)
                                    * R * kp) * np.exp(-1j * kp * R)

                                + ((x ** 2 + y ** 2) * R ** 2 * ks ** 2
                                    - 1j * (x ** 2 + y ** 2 - 2.0 * z ** 2)
                                    * R * ks) * np.exp(-1j * ks * R)
                                ) * 1j * omega[ii]

    vx = np.real(nt * np.fft.ifft(Vx, axis=0))
    vy = np.real(nt * np.fft.ifft(Vy, axis=0))
    vz = np.real(nt * np.fft.ifft(Vz, axis=0))
    return vx, vy, vz
