import argparse
import numpy as np
import os
from shutil import copyfile
import h5py as h5
import matplotlib.pyplot as plt
import lamb3D
from SeisCL import SeisCL
import homogeneous_3D
import garvin2


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
        vp_a = np.zeros([N[0], N[1], N[2]]) + vp
        vs_a = np.zeros([N[0], N[1], N[2]]) + vs
        rho_a = np.zeros([N[0], N[1], N[2]]) + rho
        taup_a = np.zeros([N[0], N[1], N[2]]) + taup
        taus_a = np.zeros([N[0], N[1], N[2]]) + taus

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
    
    print("Testing: " + testname + " ....... ", end = '')
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
        plt.suptitle("Test: " + testname + " Traces")
        ax[0].plot(data_fd[:,0], "g", label='FD')
        ax[0].plot(analytic[:, 0], "k", label='analytic')
        ax[0].plot(analytic[:, 0]-data_fd[:,0], "r", label='error')
        ax[0].set_xlabel("time (s)")
        ax[0].set_ylabel("Amplitude")
        ax[0].set_title("Offset = %f m" % offset[0])
        ax[0].legend(loc='upper right')

        mid = int(data_fd.shape[1]//2)
        ax[1].plot(data_fd[:,mid], "g")
        ax[1].plot(analytic[:, mid], "k")
        ax[1].plot(analytic[:, mid]-data_fd[:,mid], "r")
        ax[1].set_xlabel("time (s)")
        ax[1].set_ylabel("Amplitude")
        ax[1].set_title("Offset = %f m" % offset[mid])
        plt.tight_layout()

        ax[2].plot(data_fd[:,-1], "g")
        ax[2].plot(analytic[:, -1], "k")
        ax[2].plot(analytic[:, -1]-data_fd[:,-1], "r")
        ax[2].set_xlabel("time (s)")
        ax[2].set_ylabel("Amplitude")
        ax[2].set_title("Offset = %f m" % offset[-1])
        plt.tight_layout(h_pad=1, w_pad=1, pad=3)
        plt.show()

def lamb3D_test(testtype = "inline", vp=3500, vs=2000, rho=2000, taup=0, taus=0):

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
    src = lamb3D.ricker_wavelet(seis.csts["f0"], resamp*NT-1, seis.csts['dt']/resamp)
    #src = np.interp(np.arange(0, NT*resamp), np.arange(0, NT*resamp, resamp), seis.src[:,0])
    analytic = lamb3D.compute_shot(gx-sx, vp, vs, rho, seis.csts['dt']/resamp, src,
                 srctype="x", rectype="x", linedir=linedir)
    
    datafd = datafd / np.max(datafd)
    analytic = analytic[::resamp,:] / np.max(analytic)
    compare_data(datafd, analytic, gx-sx, seis.csts['dt'], "Lamb3D_"+testtype)


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

    lamb3D_test(testtype = "crossline")


