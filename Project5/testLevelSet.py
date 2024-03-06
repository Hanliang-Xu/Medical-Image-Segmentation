import json
from Project5 import * # import your custom level set and fast marching solutions
from Project.surface import * # import your previously defined surface class

def testLevelSet():
    f = open('Project5.json','rt')
    d = json.load(f)
    f.close()

    crp = np.array(d['headCT'])
    voxsz = np.array(d['voxsz'])


    fig, ax = plt.subplots(1,2)
    plt.pause(0.1)
    dmapi = np.ones(np.shape(crp))
    dmapi[2:-3,2:-3,2:-3]=-1
    ls = levelSetp5()
    params = levelSetParams(maxiter=50, visrate=1, method='CV', reinitrate=5, mindist=7, convthrsh=1e-2, mu=2, dtt=np.linspace(3,.1,50))
    dmap = ls.segment(crp, dmapi, params, ax)

    win = myVtkWin()
    s = surfacep4()
    s.createSurfaceFromVolume(-dmap, voxsz, 0)
    win.addSurf(s.verts, s.faces)
    win.start()