from scipy.spatial import Voronoi, voronoi_plot_2d
import numpy as np

def random_voronoi_cells(n_cells):
    def wrap_to_unit_box(pt):
        pt2 = pt[:]
        for i in range(2):
            if pt[i]>1:
                pt2[i]-=1
            elif pt[i]<0:
                pt2[i]+=1
        return pt2

    n_points = n_cells
    points = []
    for i in range(n_points):
        for j in range(n_points):
            points += [(1./n_points*(np.random.random_sample((1,2))+np.array([i,j])))[0]]

    # print np.array(points)
    points3x3 = np.array(points)

    for x in range(-2,3):
        for y in range(-2,3):
            if not (x==0 and y==0):
                points3x3 = np.concatenate([points3x3,points+np.array([x,y])],0)

    # print points3x3
    vor = Voronoi(points3x3)
#     voronoi_plot_2d(vor)
#     plt.plot([0,0,1,1,0],[0,1,1,0,0])
#     plt.show()
    locs = [i for i in vor.vertices if ((i[0]>0 and i[0]<1) and (i[1]>0 and i[1]<1))]
    verts = [i  for i in range(len(vor.vertices)) if ((vor.vertices[i][0]>0 and vor.vertices[i][0]<1) and (vor.vertices[i][1]>0 and vor.vertices[i][1]<1))]
    # print np.array(locs)
    # print verts

    adjs = [[] for i in locs]
    for ridge in vor.ridge_vertices:
        if (ridge[0] in verts) or (ridge[1] in verts):
    #         print ridge
            r0wrap = wrap_to_unit_box(vor.vertices[ridge[0]])
            r1wrap = wrap_to_unit_box(vor.vertices[ridge[1]])
    #         print r0wrap, r1wrap
            r0dist = 1
            r1dist = 1
            for i in range(len(locs)):
                if np.linalg.norm(locs[i]-r0wrap)<r0dist:
                    r0 = i
                    r0dist = np.linalg.norm(locs[i]-r0wrap)
                if np.linalg.norm(locs[i]-r1wrap)<r1dist:
                    r1 = i
                    r1dist = np.linalg.norm(locs[i]-r1wrap)
            adjs[r0] += [r1]
            adjs[r1] += [r0]
    adjs = [list(set(i)) for i in adjs]
    return [i*n_cells for i in locs],adjs