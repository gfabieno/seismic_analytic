import argparse
import numpy as np
import os
from shutil import copyfile
import h5py as h5
import matplotlib.pyplot as plt
import lamb3D
from SeisCL import SeisCL
from viscoelastic_3D import viscoelastic_3D
from viscoelastic_2D import viscoelastic_2D
import garvin2

def ricker_wavelet(f0, Nt, dt ):

    tmin = -1.5 / f0
    t = np.linspace(tmin, (Nt-1) * dt + tmin, num=Nt)
    ricker = ((1.0 - 2.0 * (np.pi ** 2) * (f0 ** 2) * (t ** 2))
              * np.exp(-(np.pi ** 2) * (f0 ** 2) * (t ** 2)))

    return ricker

def define_SeisCL(ND=3, dt=0.25e-03, NT=800, dh=2, f0=20, L=0, FL=[], FDORDER=4,
                  N=300, nab=112):
    seis = SeisCL()
    seis.csts['ND'] = ND
    seis.csts['N'] = np.array([N for _ in range(ND)])
    #seis.csts['no_use_GPUs'] = np.array([0])
    seis.csts['dt'] = dt
    seis.csts['NT'] = NT
    seis.csts['dh'] = dh
    seis.csts['f0'] = f0
    seis.csts['L'] = L
    seis.csts['FL'] = np.array(FL)
    seis.csts['freesurf'] = 0
    seis.csts['FDORDER'] = FDORDER
    seis.csts['MAXRELERROR'] = 1
    seis.csts['MAX'] = FDORDER
    seis.csts['abs_type'] = 2
    seis.csts['seisout'] = 1
    seis.csts['nab'] = nab
    seis.csts['abpc'] = 3  # 3 nab 112

    workdir = "./seiscl"
    if not os.path.isdir(workdir):
        os.mkdir(workdir)

    return seis

def fd_solution(seis, fileout, vp=3500, vs=2000, rho=2000, taup=0, taus=0,
                recompute=False):

    N = seis.csts['N']
    if not os.path.isfile(fileout) or recompute:
        vp_a = np.zeros(N) + vp
        vs_a = np.zeros(N) + vs
        rho_a = np.zeros(N) + rho
        taup_a = np.zeros(N) + taup
        taus_a = np.zeros(N) + taus

        seis.set_forward(seis.src_pos_all[3, :],
                         {"vp": vp_a, "rho": rho_a, "vs": vs_a,
                          "taup": taup_a, "taus": taus_a}, withgrad=False)
        seis.execute()
        copyfile(seis.workdir + "/" + seis.file_dout, fileout)
        data = seis.read_data()
    else:
        mat = h5.File(fileout, 'r')
        data = []
        for word in seis.to_load_names:
            if word + "out" in mat:
                datah5 = mat[word + "out"]
                data.append(np.transpose(datah5))

    return data

def compare_data(data_fd, analytic, offset, dt, testname, tol=10**-3, plots=1):

    err = np.sqrt(np.sum( (data_fd - analytic)**2) / np.sum(analytic**2))

    if err > tol:
        print("failed (RMSE %e)" % err)
    else:
        print("passed (RMSE %e)" % err)

    if plots:

        #Plot with shot fd, shot ana, diff
        clip = 0.1
        vmax = np.max(data_fd) * clip
        vmin = -vmax
        extent=[np.min(offset), np.max(offset), (data_fd.shape[0]-1)*dt, 0]
        fig, ax = plt.subplots(1, 3, figsize=[12, 6])
        plt.suptitle("Test: " + testname + " shot gathers")
        ax[0].imshow(data_fd, aspect='auto', vmax=vmax, vmin=vmin,
                     extent=extent, interpolation='bilinear',
                     cmap=plt.get_cmap('Greys'))
        ax[0].set_title("FD solution", fontsize=16, fontweight='bold')
        ax[0].set_xlabel("offset (m)")
        ax[0].set_ylabel("time (s)")
        ax[1].imshow(analytic, aspect='auto', vmax=vmax, vmin=vmin,
                     extent=extent, interpolation='bilinear',
                     cmap=plt.get_cmap('Greys'))
        ax[1].set_title("Analytic Solution", fontsize=16, fontweight='bold')
        ax[1].set_xlabel("offset (m)")
        ax[2].imshow(data_fd - analytic, aspect='auto', vmax=vmax, vmin=vmin,
                     extent=extent, interpolation='bilinear',
                     cmap=plt.get_cmap('Greys'))
        ax[2].set_title("Difference", fontsize=16, fontweight='bold')
        ax[2].set_xlabel("offset (m)")
        plt.tight_layout(h_pad=2, w_pad=2, pad=3)
        plt.show()

        #plot error function offset
        plt.plot(np.sum((data_fd - analytic)**2, axis=0) / np.sum(analytic**2, axis=0))
        plt.xlabel("offset (m)")
        plt.ylabel("RMSE")
        plt.title("Test: " + testname + ": error vs offset")
        plt.tight_layout()

        #plot traces short, mid and far offsets
        fig, ax = plt.subplots(3, 1, figsize=[8, 6])
        t = np.arange(0, data_fd.shape[0]*dt, dt)
        plt.suptitle("Test: " + testname + " Traces")
        ax[0].plot(t, data_fd[:,0], "g", label='FD')
        ax[0].plot(t, analytic[:, 0], "k", label='analytic')
        ax[0].plot(t, analytic[:, 0]-data_fd[:,0], "r", label='error')
        ax[0].set_xlabel("time (s)")
        ax[0].set_ylabel("Amplitude")
        ax[0].set_title("Offset = %f m" % offset[0])
        ax[0].legend(loc='upper right')

        mid = int(data_fd.shape[1]//2)
        ax[1].plot(t, data_fd[:,mid], "g")
        ax[1].plot(t, analytic[:, mid], "k")
        ax[1].plot(t, analytic[:, mid]-data_fd[:,mid], "r")
        ax[1].set_xlabel("time (s)")
        ax[1].set_ylabel("Amplitude")
        ax[1].set_title("Offset = %f m" % offset[mid])
        plt.tight_layout()

        ax[2].plot(t, data_fd[:,-1], "g")
        ax[2].plot(t, analytic[:, -1], "k")
        ax[2].plot(t, analytic[:, -1]-data_fd[:,-1], "r")
        ax[2].set_xlabel("time (s)")
        ax[2].set_ylabel("Amplitude")
        ax[2].set_title("Offset = %f m" % offset[-1])
        plt.tight_layout(h_pad=1, w_pad=1, pad=3)
        plt.show()

def lamb3D_test(testtype = "inline", vp=3500, vs=2000, rho=2000, taup=0, taus=0,
                plots=True):

    seis = define_SeisCL()
    seis.csts["freesurf"] = 1
    nbuf = seis.csts['FDORDER'] * 2
    nab = seis.csts['nab']
    dh = seis.csts['dh']
    N = seis.csts['N'][0]
    NT = seis.csts['NT']
    
    sx = (nab + nbuf) * dh
    sy = N // 2 * dh
    sz = 0 * sx
    offmin = 5 * dh
    offmax = (N - nab - nbuf) * dh - sx
    gx = np.arange(sx + offmin, sx + offmax, dh)
    gy = gx * 0 + sy
    gz = 0 * gx

    if testtype == "inline":
        srctype = 0
        rectype = 0
        linedir = "x"
    elif testtype == "crossline":
        srctype = 1
        rectype = 1
        linedir = "y"
    else:
        raise ValueError("testype must be either inline or crossline ")

    seis.src_pos = np.stack([[sx], [sy], [sz], [0], [srctype]], axis=0)
    seis.src_pos_all = seis.src_pos
    seis.fill_src()

    gsid = gz * 0
    gid = np.arange(0, len(gz))
    seis.rec_pos = np.stack([gx, gy, gz, gsid, gid, gx * 0 + rectype,
                             gx * 0, gx * 0],
                            axis=0)
    seis.rec_pos_all = seis.rec_pos

    datafd = fd_solution(seis, vp=vp, vs=vs, rho=rho, taup=taup, taus=taus,
                        fileout="lamb3D_" + testtype +".mat")
    datafd = datafd[rectype]
    
    resamp = 500
    src = ricker_wavelet(seis.csts["f0"], resamp*NT-1, seis.csts['dt']/resamp)
    analytic = lamb3D.compute_shot(gx-sx, vp, vs, rho, seis.csts['dt']/resamp, src,
                 srctype="x", rectype="x", linedir=linedir)
    
    datafd = datafd / np.max(datafd)
    analytic = analytic[::resamp,:] / np.max(analytic)
    compare_data(datafd, analytic, gx-sx, seis.csts['dt'], "Lamb3D_"+testtype,
                 plots=plots)


def garvin2D_test(vp=3500, vs=2000, rho=2000, taup=0, taus=0,
                  plots=True):
    seis = define_SeisCL(ND=2)
    seis.csts["freesurf"] = 1
    nbuf = seis.csts['FDORDER'] * 2
    nab = seis.csts['nab']
    dh = seis.csts['dh']
    N = seis.csts['N'][0]
    NT = seis.csts['NT']

    sx = (nab + nbuf) * dh
    sy = 0
    sz = dh * 10
    offmin = 5 * dh
    offmax = (N - nab - nbuf) * dh - sx
    gx = np.arange(sx + offmin, sx + offmax, dh)
    gy = gx * 0
    gz = gx * 0 + dh * 10

    srctype = 100
    rectype = 0

    seis.src_pos = np.stack([[sx], [sy], [sz], [0], [srctype]], axis=0)
    seis.src_pos_all = seis.src_pos
    seis.fill_src()

    gsid = gz * 0
    gid = np.arange(0, len(gz))
    seis.rec_pos = np.stack([gx, gy, gz, gsid, gid, gx * 0 + rectype,
                             gx * 0, gx * 0], axis=0)
    seis.rec_pos_all = seis.rec_pos

    datafd = fd_solution(seis, vp=vp, vs=vs, rho=rho, taup=taup, taus=taus,
                         fileout="Garvin2D.mat")
    datafd = datafd[rectype]

    resamp = 50
    src = ricker_wavelet(seis.csts["f0"], resamp * NT - 1,
                                 seis.csts['dt'] / resamp)
    analytic = garvin2.compute_shot(gx - sx + dh/2, vp, vs, rho,
                                    seis.csts['dt'] / resamp, src, rectype="x",
                                    zsrc=sz, zrec=gz[0])

    #datafd = datafd / np.max(datafd)
    #analytic = analytic[::resamp, :] / np.max(analytic)

    datafd = datafd / np.sqrt(np.sum(datafd**2))
    analytic = analytic[::resamp, :]
    analytic = analytic / np.sqrt(np.sum(analytic** 2))

    compare_data(datafd, analytic, gx - sx+ dh/2, seis.csts['dt'], "Garvin2D",
                 plots=plots)

def homogeneous3D_test(testname, vp=3500, vs=2000, rho=2000, taup=0, taus=0,
                       ND=3, FDORDER=4, plots=True, testtype="inline"):

    seis = define_SeisCL(FDORDER=FDORDER, ND=ND)
    seis.csts["freesurf"] = 0
    nbuf = seis.csts['FDORDER'] * 2
    nab = seis.csts['nab']
    dh = seis.csts['dh']
    N = seis.csts['N'][0]
    NT = seis.csts['NT']
    if taup != 0 or taus != 0:
        seis.csts['L'] = 1
        seis.csts['FL'] = np.array([seis.csts["f0"]])
        omega0 = 2 * np.pi * seis.csts['FL']
    else:
        omega0 = []

    if testtype == "inline":
        sx = N // 2 * dh
        sy = N // 2 * dh
        sz = (nab + nbuf) * dh
        offmin = 5 * dh
        offmax = (N - nab - nbuf) * dh - sz
        gz = np.arange(sz + offmin, sz + offmax, dh)
        gx = gz * 0 + N // 2 * dh
        gy = gz * 0 + N // 2 * dh
        offsets = gz-sz
    elif testtype == "crossline":
        sx = (nab + nbuf) * dh
        sy = N // 2 * dh
        sz = N // 2 * dh
        offmin = 5 * dh
        offmax = (N - nab - nbuf) * dh - sx
        gx = np.arange(sx + offmin, sx + offmax, dh)
        gy = gx * 0 + N // 2 * dh
        gz = gx * 0 + N // 2 * dh
        offsets = gx-sx
    else:
        raise ValueError("testype must be either inline or crossline ")

    seis.src_pos = np.stack([[sx], [sy], [sz], [0], [2]], axis=0)
    seis.src_pos_all = seis.src_pos
    seis.fill_src()

    gsid = gz * 0
    gid = np.arange(0, len(gz))
    seis.rec_pos = np.stack([gx, gy, gz, gsid, gid, gx * 0 + 2,
                             gx * 0, gx * 0], axis=0)
    seis.rec_pos_all = seis.rec_pos
    datafd = fd_solution(seis, vp=vp, vs=vs, rho=rho, taup=taup, taus=taus,
                         fileout=testname+".mat")
    datafd = datafd[-1]

    src = ricker_wavelet(seis.csts["f0"], 2*NT, seis.csts['dt'])
    rec_pos = [[gx[ii]-sx, gy[ii]-sy, gz[ii]-sz] for ii in range(0, len(gx))]

    if ND == 3:
        analytic = viscoelastic_3D(vp, vs, rho, taup, taus,
                                   omega0, seis.csts['dt'],
                                   rec_pos, src)
    else:
        analytic = viscoelastic_2D(vp, vs, rho, taup, taus,
                                   omega0, seis.csts['dt'],
                                   rec_pos, src)
    analytic = analytic[-1][:NT, :]
    datafd = datafd / np.sqrt(np.sum(datafd ** 2))
    analytic = analytic / np.sqrt(np.sum(analytic ** 2))

    compare_data(datafd, analytic, offsets, seis.csts['dt'], testname,
                 plots=plots)



if __name__ == "__main__":

    # Initialize argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--test",
                        type=str,
                        default='all',
                        help="Name of the test to run, default to all"
                        )
    parser.add_argument("--plot",
                        type=int,
                        default=0,
                        help="Plot the test results (1) or not (0). Default: 0."
                        )

    # Parse the input for training parameters
    args, unparsed = parser.parse_known_args()

    name = "Lamb3D_inline"
    if args.test == name or args.test == "all":
        print("Testing: " + name + " ....... ", end='')
        lamb3D_test(testtype="inline", plots=args.plot)

    name = "Lamb3D_crossline"
    if args.test == name or args.test == "all":
        print("Testing: " + name + " ....... ", end='')
        lamb3D_test(testtype="crossline", plots=args.plot)

    name = "Garvin_2D"
    if args.test == name or args.test == "all":
        print("Testing: " + name + " ....... ", end='')
        garvin2D_test(plots=args.plot)

    name = "elastic_3D_inline"
    if args.test == name or args.test == "all":
        print("Testing: " + name + " ....... ", end='')
        homogeneous3D_test(testname=name, testtype="inline", plots=args.plot)

    name = "elastic_3D_crossline"
    if args.test == name or args.test == "all":
        print("Testing: " + name + " ....... ", end='')
        homogeneous3D_test(testname=name, testtype="crossline", plots=args.plot)

    name = "viscoelastic_3D_inline"
    if args.test == name or args.test == "all":
        print("Testing: " + name + " ....... ", end='')
        homogeneous3D_test(testname=name, testtype="inline", plots=args.plot,
                           taup=0.2, taus=0.2)

    name = "viscoelastic_3D_crossline"
    if args.test == name or args.test == "all":
        print("Testing: " + name + " ....... ", end='')
        homogeneous3D_test(testname=name, testtype="crossline", plots=args.plot,
                           taup=0.2, taus=0.2)

    name = "elastic_2D_crossline"
    if args.test == name or args.test == "all":
        print("Testing: " + name + " ....... ", end='')
        homogeneous3D_test(testname=name, ND=2,
                           testtype="crossline", plots=args.plot)

    name = "elastic_2D_inlineline"
    if args.test == name or args.test == "all":
        print("Testing: " + name + " ....... ", end='')
        homogeneous3D_test(testname=name, ND=2,
                           testtype="inline", plots=args.plot)

    name = "viscoelastic_2D_crossline"
    if args.test == name or args.test == "all":
        print("Testing: " + name + " ....... ", end='')
        homogeneous3D_test(testname=name, ND=2,
                           testtype="crossline", plots=args.plot,
                           taup=0.2, taus=0.2)

