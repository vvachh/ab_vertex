import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
from numpy import pi
from mpl_toolkits.mplot3d import Axes3D
import mayavi.mlab as mlab
import pickle

def same(x,y):
    return (x and y) or ((not x) and (not y))
def switchaforb(lis,a,b):
    lis[lis.index(a)] = b

class VertexModelSphere:

    def __init__(self, locs, adjs, radius=1., p0=1., r=1.):
        self.v = [i.copy() for i in locs]
        self.adj = [sorted(i[:]) for i in adjs]
        self.radius = radius
        self.v_cart = map(self.spherical_to_cartesian,self.v)
        self.A0 = 4.*pi*radius*radius/len(self.v)
        self.p0 = p0
        self.r = r
        self.edges = [self.make_edges(i) for i in range(len(locs))]
        self.edgelens = [self.make_edgelens(i) for i in range(len(self.v))]
        self.faces = self.find_all_faces()
        self.faceadj = self.make_faceadj()
        self.faceareas = map(self.find_facearea, range(len(self.faces)))
        self.perims = map(self.make_perim, range(len(self.faces)))
        
        self.energy = self.total_mechanical_energy()

    #I can't handle spherical coordinates so here
    def spherical_to_cartesian(self,sph):
        r,theta,phi = self.radius,sph[0],sph[1]
        x = r*np.sin(theta)*np.cos(phi)
        y = r*np.sin(theta)*np.sin(phi)
        z = r*np.cos(theta)

        return np.array([x,y,z])
    def cartesian_to_spherical(self,cart):
        x,y,z = cart[0],cart[1],cart[2]

        phi = np.arctan2(y,x)
        theta = np.arctan2(np.sqrt(x*x+y*y),z)

        return np.array([theta,phi])
    
    #Getters and setters
    def get_vert(self,idx):
        return self.v[idx].copy()
    def get_vert_cart(self,idx):
        return self.v_cart[idx].copy()
    def get_adj(self,idx):
        return self.adj[idx][:]
    def neighbors(self,idx):
        # returns neighbors of the vertex idx: list of positions.
        adjacency = self.adj[idx]
        return map(self.get_vert, adjacency)
    def make_edge(self,idx1,idx2):
        # returns the vector which points in the direction of idx1 to idx2 and has length equal to the edge length.
        # Use to correct for periodic boundary conditions.
        loc1 = self.get_vert_cart(idx1)
        loc2 = self.get_vert_cart(idx2)
        edge = loc2-loc1
        return edge
    def make_edges(self,idx):
        # return all the edges from a single vertex
        loc = self.get_vert_cart(idx)
        neighbors = self.get_adj(idx)
        return map(lambda x: self.make_edge(idx,x), neighbors)
    def get_edge(self,idx1,idx2):
        j = self.adj[idx1].index(idx2)
        if j == None:
            print 'edge doesn\'t exist'
            return
        return self.edges[idx1][j]
    def get_edges(self,idx):
        edgecopy = [self.edges[idx][i].copy() for i in range(3)]
        return edgecopy

    def make_edgelen(self,idx1,idx2):
        #find the edge length between idx1 and idx2, using the shortest great-circle distance between the points.
        loc1 = self.get_vert(idx1)
        loc2 = self.get_vert(idx2)
        diff = np.abs(loc1-loc2)
        dLat = diff[0]; dLong = diff[1]
        lat1 = pi/2-loc1[0]
        lat2 = pi/2-loc2[0]
        dsig = 2*np.arcsin(np.sqrt(
            np.power(np.sin(dLat/2),2)+
            np.cos(lat1)*np.cos(lat2)*np.power(np.sin(dLong/2),2)
        ))
        return self.radius*dsig
    def make_edgelens(self,idx):
        return [self.make_edgelen(idx,idx2) for idx2 in self.adj[idx]]
    def get_edgelen(self,idx1,idx2):
        return self.edgelens[idx1][self.adj[idx1].index(idx2)]
    
    def find_face(self,v1,v2,v3):
        ## find the face that includes the three indices idx1-idx2-idx3
        edges2 = self.get_tangent_vecs(v2)
        
        idx1_2 = self.get_adj(v2).index(v1)
        idx3_2 = self.get_adj(v2).index(v3)
        idxn_2 = [i for i in range(3) if (i!=idx1_2 and i!=idx3_2)][0]
        
        #cw or ccw
        ccw = False
        angles2 = [np.arctan2(x[1],x[0]) for x in edges2]
        #print angles2
        if same((angles2[idx3_2]>angles2[idxn_2]),(angles2[idxn_2]>angles2[idx1_2])):
            #print 'between'
            if angles2[idx3_2]>angles2[idx1_2]:
                ccw = False
            else:
                ccw = True
        else:
            if angles2[idx3_2]>angles2[idx1_2]:
                ccw = True
            else:
                ccw = False
        
        #print ccw
        #travel around the face and get the vertices
        face = [v1, v2, v3]
        prev_vertex = v2
        current_vertex = v3

        while current_vertex!=v1:
            #get the other two neighbors
            adj = self.get_adj(current_vertex)
            edg = self.get_tangent_vecs(current_vertex)
            angs = [np.arctan2(x[1],x[0]) for x in edg]
            
            idx_next = range(3)
            idx_prev = adj.index(prev_vertex)
            #print adj
            idx_next.remove(idx_prev)
            
            if same((angs[idx_next[0]]>angs[idx_prev]),(angs[idx_prev]>angs[idx_next[1]])):
                if angs[idx_next[0]]>angs[idx_next[1]]:
                    if ccw:
                        face+=[adj[idx_next[0]]]
                    else:
                        face+=[adj[idx_next[1]]]
                else:
                    if not ccw:
                        face+=[adj[idx_next[0]]]
                    else:
                        face+=[adj[idx_next[1]]]
            else:
                if angs[idx_next[0]]>angs[idx_next[1]]:
                    if not ccw:
                        face+=[adj[idx_next[0]]]
                    else:
                        face+=[adj[idx_next[1]]]
                else:
                    if ccw:
                        face+=[adj[idx_next[0]]]
                    else:
                        face+=[adj[idx_next[1]]]
            
            prev_vertex = current_vertex
            current_vertex = face[-1]
            #print current_vertex
        if not ccw:
            return face[:-1]
        else:
            return face[-1:0:-1]
    def get_face(self,fidx):
        return self.faces[fidx][:]
    def get_face_which_contains(self,v1,v2,v3):
        #return the actual face index containing those three vertices in order
        facetest = self.find_face(v1,v2,v3)
        for f in range(len(self.faces)):
            if set(facetest)==set(self.faces[f]):
                return f
        print 'Not on the same face, I think'

    def replace_part_of_face(self,faceidx,find,replace):
        start = None
        end = None
        for i in range(len(self.faces[faceidx])):
            if self.faces[faceidx][i]==find[0]:
                start = i
            elif self.faces[faceidx][i]==find[-1]:
                end = i
        if end<start:
            newface = self.faces[faceidx][end+1:start]
        else:
            newface = self.faces[faceidx][end+1:]+self.faces[faceidx][:start]
        newface += replace
        self.faces[faceidx] = newface

    def find_all_faces(self):
        #goes along each edge to search for all the faces. 
        #Returns a list of faces, each represented by a list of vertices
        faces = []
        facesets = []
        for v in range(len(self.v)):
            n = self.adj[v]
            for w in n:
                n2 = [i for i in self.adj[w] if i!=v]
                #print v,w,n2[0]
                #print v,w,n2[1]
                face0 = self.find_face(v,w,n2[0])
                face1 = self.find_face(v,w,n2[1])
                if set(face0) not in facesets:
                    faces+= [face0]
                    facesets+=[set(face0)]
                if set(face1) not in facesets:
                    faces+= [face1]
                    facesets+=[set(face1)]
        return faces
    def make_faceadj(self):
        #makes a list of face indices by vertex: each vertex is part of three faces.
        faceadj = [[] for i in range(len(self.v))]
        for faceidx in range(len(self.faces)):
            for v in self.faces[faceidx]:
                faceadj[v]+=[faceidx]
        return faceadj
    def find_facearea(self,idx):
        #find the area of face idx
        S = 0
        for i in range(-1,len(self.faces[idx])-1):
            S += self.angle_between(self.faces[idx][i-1],self.faces[idx][i],self.faces[idx][i+1])
        return (S-(len(self.faces[idx])-2)*pi)*self.radius*self.radius

    def make_perim(self,fidx):
        #compute the perimeter of the spherical polygon
        face = self.get_face(fidx)
        perim = 0
        for i in range(len(face)):
            perim += self.get_edgelen(face[i-1],face[i])
        return perim
    
    #moving vertices
    def move_vertex(self,idx,new_pos):
        self.v[idx] = new_pos
        self.v_cart[idx] = self.spherical_to_cartesian(new_pos)
        #update edgelens
        self.edgelens[idx] = self.make_edgelens(idx)
        self.edges[idx] = self.make_edges(idx)
        for nb in self.adj[idx]:
            self.edgelens[nb] = self.make_edgelens(nb)
            self.edges[nb] = self.make_edges(nb)
        #update faceareas
        for f in self.faceadj[idx]:
            self.faceareas[f] = self.find_facearea(f)
            self.perims[f] = self.make_perim(f)
    def move_vertex_by(self,idx,increment):
        new_pos = self.v[idx]+increment
        self.v[idx] = new_pos
        self.v_cart[idx] = self.spherical_to_cartesian(new_pos)
        #update edgelens
        self.edgelens[idx] = self.make_edgelens(idx)
        self.edges[idx] = self.make_edges(idx)
        for nb in self.adj[idx]:
            self.edgelens[nb] = self.make_edgelens(nb)
            self.edges[nb] = self.make_edges(nb)
        #update faceareas
        for f in self.faceadj[idx]:
            self.faceareas[f] = self.find_facearea(f)
            self.perims[f] = self.make_perim(f)
    def switch_vertices(self, idx1,idx2):
        self.v[idx1],self.v[idx2] = self.v[idx2],self.v[idx1]
        
        self.adj[idx1],self.adj[idx2] = self.adj[idx2],self.adj[idx1]
        self.adj[idx1][self.adj[idx1].index(idx1)] = idx2
        self.adj[idx2][self.adj[idx2].index(idx2)] = idx1
        self.adj[idx1], self.adj[idx2] = sorted(self.adj[idx1]), sorted(self.adj[idx2])
        for i in self.adj[idx1]:
            if i!=idx2:
                self.adj[i][self.adj[i].index(idx2)] = idx1
        for i in self.adj[idx2]:
            if i!=idx1:
                self.adj[i][self.adj[i].index(idx1)] = idx2

        self.v_cart[idx1],self.v_cart[idx2] = self.v_cart[idx2],self.v_cart[idx1]
        
        self.edges[idx1] = self.make_edges(idx1)
        self.edges[idx2] = self.make_edges(idx2)
        for i in self.adj[idx1]:
            self.edges[i] = self.make_edges(i)
        for i in self.adj[idx2]:
            self.edges[i] = self.make_edges(i)

        self.edgelens[idx1] = self.make_edgelens(idx1)
        self.edgelens[idx2] = self.make_edgelens(idx2)
        for i in self.adj[idx1]:
            self.edgelens[i] = self.make_edgelens(i)
        for i in self.adj[idx2]:
            self.edgelens[i] = self.make_edgelens(i)

        self.faceadj[idx1],self.faceadj[idx2] = self.faceadj[idx2],self.faceadj[idx1]
        
        affected_faces = list(set(self.faceadj[idx1]+self.faceadj[idx2]))
        for i in range(len(affected_faces)):
            # print self.faces[affected_faces[i]]
            f = self.faces[affected_faces[i]]
            if (idx1 in f) and (idx2 in f):
                #switch the order
                # print 'both are here!'
                i1 = f.index(idx1)
                i2 = f.index(idx2)
                # print idx1,'is at',i1,'and',idx2,'is at',i2
                f[i1],f[i2] = idx2,idx1
            elif idx2 in f:
                f[f.index(idx2)] = idx1
            elif idx1 in f:
                f[f.index(idx1)] = idx2
            else:
                print 'huh?'
            # print self.faces[affected_faces[i]]
        

    #T1 topological transitions
    def t1_trans(self,idx1,idx2):
        edge = self.get_edge(idx1,idx2)
        loc1_cart = self.get_vert_cart(idx1)
        loc1 = self.get_vert(idx1)
        loc2 = self.get_vert(idx2)
        loc_new = self.cartesian_to_spherical(loc1_cart+edge/2.)
        
        self.v[idx1] = loc_new.copy()
        self.v[idx2] = loc_new.copy()
        neighbors1 = [x for x in self.get_adj(idx1) if x!=idx2]
        neighbors2 = [x for x in self.get_adj(idx2) if x!=idx1]
        

        
        #order the neighbors clockwise:
        n = neighbors1 + neighbors2
        self.v[idx1] = loc_new.copy()
        self.v[idx2] = loc_new.copy()
        nlocs = [self.tangent_vec(idx1,i) for i in neighbors1] + [self.tangent_vec(idx2,i) for i in neighbors2]
        self.v[idx1] = loc1
        self.v[idx2] = loc2
        nangs = [np.arctan2(i[1],i[0]) for i in nlocs]
        sort_indices = sorted(range(len(nangs)),key = lambda x: nangs[x])
        n_sorted = [n[x] for x in sort_indices]
        nlocs_sorted = [nlocs[x] for x in sort_indices]
        for shift in range(4):
            if (n_sorted[-shift] in neighbors1) and (n_sorted[-shift+1] in neighbors1):
                n_sorted = n_sorted[-shift:] + n_sorted[0:4-shift]
                break 
        v10 = n_sorted[0]
        v11 = n_sorted[1]
        v20 = n_sorted[2]
        v21 = n_sorted[3]
        #switch neighbors:
        A = self.get_face_which_contains(n_sorted[0],idx1,n_sorted[1])
        B = self.get_face_which_contains(idx2,idx1,n_sorted[1])
        C = self.get_face_which_contains(idx1,idx2,n_sorted[3])
        D = self.get_face_which_contains(n_sorted[2],idx2,n_sorted[3])

        # print A,B,C,D
        self.v[idx1] = loc_new.copy()
        self.v[idx2] = loc_new.copy()
        self.v_cart[idx1] = self.spherical_to_cartesian(loc_new)
        self.v_cart[idx2] = self.spherical_to_cartesian(loc_new)
        
        #update adjacencies, enforcing that the entries of self.adj are sorted.
        neighbors1 = sorted([idx2,n_sorted[3],n_sorted[0]])
        neighbors2 = sorted([idx1,n_sorted[1],n_sorted[2]])
        self.adj[idx1] = neighbors1
        self.adj[idx2] = neighbors2

        self.adj[n_sorted[1]][self.adj[n_sorted[1]].index(idx1)] = idx2
        self.adj[n_sorted[1]] = sorted(self.adj[n_sorted[1]])

        self.adj[n_sorted[3]][self.adj[n_sorted[3]].index(idx2)] = idx1
        self.adj[n_sorted[3]] = sorted(self.adj[n_sorted[3]])

        #update edges
        self.edges[idx1] = self.make_edges(idx1)
        self.edges[idx2] = self.make_edges(idx2)
        self.edges[n_sorted[0]] = self.make_edges(n_sorted[0])
        self.edges[n_sorted[1]] = self.make_edges(n_sorted[1])
        self.edges[n_sorted[2]] = self.make_edges(n_sorted[2])
        self.edges[n_sorted[3]] = self.make_edges(n_sorted[3])

        #update edgelens
        self.edgelens[idx1] = self.make_edgelens(idx1)
        self.edgelens[idx2] = self.make_edgelens(idx2)
        self.edgelens[n_sorted[0]] = self.make_edgelens(n_sorted[0])
        self.edgelens[n_sorted[1]] = self.make_edgelens(n_sorted[1])
        self.edgelens[n_sorted[2]] = self.make_edgelens(n_sorted[2])
        self.edgelens[n_sorted[3]] = self.make_edgelens(n_sorted[3])
        
        #update faces and faceadj
        self.replace_part_of_face(A,[v11,idx1,v10],[v11,idx2,idx1,v10])
        self.replace_part_of_face(B, [v20,idx2,idx1,v11],[v20,idx2,v11])
        #update locations
        self.replace_part_of_face(C,[v10,idx1,idx2,v21],[v10,idx1,v21])
        self.replace_part_of_face(D, [v21,idx2,v20],[v21,idx1,idx2,v20])
        
        #update faceadj
        switchaforb(self.faceadj[idx1], B, D)
        switchaforb(self.faceadj[idx2], C, A)
        
        #update faceareas
        self.faceareas[A] = self.find_facearea(A)
        self.faceareas[B] = self.find_facearea(B)
        self.faceareas[C] = self.find_facearea(C)
        self.faceareas[D] = self.find_facearea(D)

        #update perims
        self.perims[A] = self.make_perim(A)
        self.perims[B] = self.make_perim(B)
        self.perims[C] = self.make_perim(C)
        self.perims[D] = self.make_perim(D)
        
        #return vectors pointing away from the junction
        dis1 = np.cross(loc1_cart, edge)/2.
        dis1 = dis1/norm(dis1)*norm(edge)/2.
        dis2 = np.cross(loc1_cart, -edge)/2.
        dis2 = dis2/norm(dis2)*norm(edge)/2.

        self.move_vertex(idx1,self.cartesian_to_spherical(self.v_cart[idx1]+dis1))
        self.move_vertex(idx2,self.cartesian_to_spherical(self.v_cart[idx2]+dis2))
        
        return A,B,C,D


    #mechanical energy
    def total_mechanical_energy(self):
        A0 = self.A0
        p0 = self.p0
        r = self.r
        energy = 0
        for f in range(len(self.faces)):
            A = self.faceareas[f]
            p = self.perims[f]
            A_tilde = A/A0
            p_tilde = p/np.power(A0,0.5)
            energy += (A_tilde-1)*(A_tilde-1) + (p_tilde-p0)*(p_tilde-p0)/r
        return energy
    def mech_energy_vertex(self,idx):
        #just compute the relevant energy terms related to the vertex idx
        A0 = self.A0
        p0 = self.p0
        r = self.r
        energy = 0
        for f in self.faceadj[idx]:
            A = self.faceareas[f]
            p = self.perims[f]
            A_tilde = A/A0
            p_tilde = p/np.power(A0,0.5)
            energy += (A_tilde-1)*(A_tilde-1) + (p_tilde-p0)*(p_tilde-p0)/r
        return energy

    #gradient descent
    def gradient_descent_one_vertex(self,idx,thetastep,phistep):
        #very elementary algorithm: just moving around in the 8-neighborhood defined by thetastep,phistep
        best_theta = 0
        best_phi = 0
        bestE = self.mech_energy_vertex(idx)
        for dtheta in [-thetastep,0,thetastep]:
            for dphi in [-phistep,0,phistep]:
                increment = np.array([dtheta,dphi])
                self.move_vertex_by(idx,increment)
                E = self.mech_energy_vertex(idx)
                if E<bestE:
                    bestE = E
                    best_theta = dtheta
                    best_phi = dphi
                self.move_vertex_by(idx,-increment)
        self.move_vertex_by(idx,np.array([best_theta,best_phi]))
    def gradient_descent_step(self,thetastep,phistep):
        energy_before = self.total_mechanical_energy()
        for idx in range(len(self.v)):
            self.gradient_descent_one_vertex(idx,thetastep,phistep)
        self.energy = self.total_mechanical_energy()
        step = self.energy - energy_before
        return step
    def gradient_descent_no_t1(self,thetastep,phistep):
        step = -1
        while step!=0:
            step = self.gradient_descent_step(thetastep,phistep)
            print 'energy step:',step
        print 'done'

    def evaluate_t1_trans(self,i,k):
        # carry out a passive T1 trans and do it if it decreases energy
        edgelens_copy = [x[:] for x in self.edgelens]
        faces_copy = [x[:] for x in self.faces]
        faceadj_copy = [x[:] for x in self.faceadj]
        faceareas_copy = self.faceareas[:]
        perims_copy = self.perims[:]
        adj_copy = [x[:] for x in self.adj]
        v_copy = [x.copy() for x in self.v]
        v_cart_copy = [x.copy() for x in self.v_cart]
        
        E_notrans = self.total_mechanical_energy()

        self.t1_trans(i,k)

        E_withtrans = self.total_mechanical_energy()

        if E_withtrans>E_notrans:
            self.edgelens = edgelens_copy
            self.faces = faces_copy
            self.faceadj = faceadj_copy
            self.faceareas = faceareas_copy
            self.perims = perims_copy
            self.adj = adj_copy
            self.energy = E_notrans
            self.v = v_copy
            self.v_cart = v_cart_copy
            return 0
        self.energy = E_withtrans
        return 1
    def gradient_descent_with_t1(self, thetastep, phistep, l_crit):
        step = -1
        total_trans = 0
        while step!=0:
            step = self.gradient_descent_step(thetastep,phistep)
            for i in range(len(self.v)):
                for j in range(3):
                    if self.edgelens[i][j]<l_crit:
                        E_i = self.energy
                        k = self.adj[i][j]
                        total_trans += self.evaluate_t1_trans(i,k)
            print 'energy step:',step
        print 'done'
        print 'total T1 transitions:',total_trans

    #spherical geometry things
    def rotate_wrt(self,idx1,idx2):
        #return the smallest-magnitude [theta,phi] increment that gets you from idx1 to idx2
        edge = self.make_edge(idx1,idx2)
        loc1 = self.get_vert(idx1)
        theta = loc1[0]
        phi = loc1[1]
        #rotation about y axis
        undo_phi = np.array([
            [np.cos(-phi),-np.sin(-phi),0],
            [np.sin(-phi),np.cos(-phi),0],
            [0,0,1]])
        #rotation about z axis
        undo_theta = np.array([
            [np.cos(-theta),0,np.sin(-theta)],
            [0,1,0],
            [-np.sin(-theta),0,np.cos(-theta)]])
        loc2 = self.get_vert_cart(idx2)
        loc2_rot = np.dot(undo_theta,np.dot(undo_phi,loc2))
        return loc2_rot
    def tangent_vec(self,idx1,idx2):
        loc2 = self.rotate_wrt(idx1,idx2)
        ori = np.array([0,0,1])
        tanvec = np.cross(ori,np.cross(loc2,ori))
        if tanvec[2]>0.0000001:
            print 'something is quite wrong here'
        return tanvec[0:2]
    def get_tangent_vecs(self,idx):
        #get the tangent vectors for all the neighbors from idx
        tvecs = []
        for idx2 in self.get_adj(idx):
            tvecs += [self.tangent_vec(idx,idx2)]
        # print tvecs
        return tvecs
    def angle_between(self,idx1,idx2,idx3):
        # computes the angle on the surface of the sphere between idx1-idx2-idx3, going a particular way.
        tv1 = self.tangent_vec(idx2,idx1)
        tv2 = self.tangent_vec(idx2,idx3)
        if np.cross(tv1,tv2)<0:
            return np.arccos(np.dot(tv1,tv2)/(norm(tv1)*norm(tv2)))
        else:
            return 2*pi - np.arccos(np.dot(tv1,tv2)/(norm(tv1)*norm(tv2)))
    def rotate_to_focus(self,idx):
        #rotate all vertices such that idx is at the top of the sphere.
        for i in range(len(self.v)):
            if i!=idx:
                self.v_cart[i] = self.rotate_wrt(idx,i)
                self.v[i] = self.cartesian_to_spherical(self.v_cart[i])
        self.v_cart[idx] = np.array([0,0,1])
        self.v[idx] = np.array([0,0])
        for i in range(len(self.v)):
            self.edges[i] = self.make_edges(i)
            
    #measurements
    def find_rosettes(self,l_crit):
        #find groups of vertices all connected by edges of length <l_crit
        #returns a list of (lists of vertices in a rosette)
        rosettes = []
        for i in range(len(self.v)):
            for j in range(3):
                if self.edgelens[i][j]<l_crit:
                    k = self.adj[i][j]
                    found_rosette = False
                    for rosette in rosettes:
                        if i in rosette:
                            found_rosette = True
                            if k not in rosette:
                                rosette+= [k]
                        elif k in rosette:
                            found_rosette = True
                            if i not in rosette:
                                rosette+= [i]
                    if not found_rosette:
                        rosettes+= [[i,k]]
        return rosettes
    def rosette_degree(self,rosette):
        #return the degree of the rosette: how many faces meet at the rosette?
        faces = list(set(np.concatenate([np.concatenate([self.faces[faceidx] for faceidx in self.faceadj[vidx]]) for vidx in rosette])))
        return len(faces)
    def analyze_rosettes(self,l_crit,showplot=True):
        rosettes = self.find_rosettes(l_crit)
        degrees = [self.rosette_degree(ros) for ros in rosettes]
        
        if showplot:
            hist, bins = np.histogram(degrees,bins='auto')
            width = 0.7 * (bins[1] - bins[0])
            center = (bins[:-1] + bins[1:]) / 2
            plt.bar(center, hist, align='center', width=width)
            plt.plot([self.radius/(self.radius+self.thickness), self.radius/(self.radius+self.thickness)], [0,max(hist)])
            plt.show()
        return rosettes,degrees

    #drawing things    
    def draw_sphere(self,fig,color=(0,0,1), alpha=0.5):
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        radius = self.radius

        x = radius * np.outer(np.cos(u), np.sin(v))
        y = radius * np.outer(np.sin(u), np.sin(v))
        z = radius * np.outer(np.ones(np.size(u)), np.cos(v))
        #for i in range(2):
        #    ax.plot_surface(x+random.randint(-5,5), y+random.randint(-5,5), z+random.randint(-5,5),  rstride=4, cstride=4, color='b', linewidth=0, alpha=0.5)

        mlab.mesh(x, y, z,  color=color,figure=fig,opacity=alpha)
    def draw_greatcirc(self,pos1,pos2,fig):
        #not really true: doesn't draw great circle arcs.
        pos1 = self.cartesian_to_spherical(pos1)
        pos2 = self.cartesian_to_spherical(pos2)
        thetas = np.linspace(pos1[0],pos2[0],20)
        phis = np.linspace(pos1[1],pos2[1],20)
        coords = zip(thetas,phis)
        carts = np.array(map(self.spherical_to_cartesian, coords))

        mlab.plot3d(carts[:,0],carts[:,1],carts[:,2],figure=fig)
    def draw_model(self,figi=None,color=(0,0,1),alpha=0.5):
        if figi==None:
            fig = mlab.figure()
        else:
            fig = figi
        vtxs = np.array(self.v_cart)
        mlab.points3d(vtxs[:,0],vtxs[:,1],vtxs[:,2],scale_factor=0.05)
        for i in range(len(self.v)):
            pt1 = self.v_cart[i]
            mlab.text(pt1[0],pt1[1],str(i),z=pt1[2])
            for j in self.edges[i]:
                mlab.plot3d([pt1[0],pt1[0]+j[0]],
                    [pt1[1],pt1[1]+j[1]],
                    [pt1[2],pt1[2]+j[2]],figure=fig,tube_radius=None)
        self.draw_sphere(fig,color=color, alpha=alpha)
        if figi==None:
            mlab.show()

    #saving and loading models
    def cache_params(self):
        edges_copy = [x[:] for x in self.edges]
        edgelens_copy = [x[:] for x in self.edgelens]
        faces_copy = [x[:] for x in self.faces]
        faceadj_copy = [x[:] for x in self.faceadj]
        faceareas_copy = self.faceareas[:]
        perims_copy = self.perims[:]
        adj_copy = [x[:] for x in self.adj]
        v_copy = [x.copy() for x in self.v]
        v_cart_copy = [x.copy() for x in self.v_cart]

        return (edges_copy, edgelens_copy, faces_copy, faceadj_copy, faceareas_copy, perims_copy, adj_copy, v_copy, v_cart_copy)

    def load_params(self, params):
        (edges_copy, edgelens_copy, faces_copy, faceadj_copy, faceareas_copy, perims_copy, adj_copy, v_copy, v_cart_copy) = params
        self.edges = edges_copy
        self.edgelens = edgelens_copy
        self.faces = faces_copy
        self.faceadj = faceadj_copy
        self.faceareas = faceareas_copy
        self.perims = perims_copy
        self.adj = adj_copy
        self.v = v_copy
        self.v_cart = v_cart_copy

    def save_to_file(self, filename):
        #pickle v, adjs, radius, p0, and r.
        #importantly, we're not pickling the actual VertexModel instance so that I can change object methods without messing everything up

        with open(filename,'wb') as pickle_file:
            data = {'locs':self.v, 'adjs':self.adj, 'radius':self.radius, 'p0':self.p0, 'r':self.r}
            pickle.dump(data, pickle_file)
    @staticmethod
    def load_from_file(filename):
        with open(filename,'rb') as pickle_file:
            data = pickle.load(pickle_file)
        vm = VertexModelSphere(data['locs'],data['adjs'],radius=data['radius'],p0=data['p0'], r=data['r'])
        return vm
