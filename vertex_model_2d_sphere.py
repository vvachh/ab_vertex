import numpy as np
import matplotlib.pyplot as plt

class VertexModelSphere:

    def __init__(self, locs, adjs, radius=1., A0=1./12, p0=1., r=1.):
        self.v = [i.copy() for i in locs]
        self.v_cart = [self.spherical_to_cartesian for i in self.v]
        self.adj = [i[:] for i in adjs]
        self.radius = radius
        self.A0 = A0
        self.p0 = p0
        self.r = r
        self.edges = [self.make_edges(i) for i in range(len(locs))]
        self.edgelens = [map(np.linalg.norm,self.make_edges(i)) for i in range(len(locs))]
        
        self.faces = self.find_all_faces()
        self.faceadj = self.make_faceadj()
        self.faceareas = map(self.get_facearea, range(len(self.faces)))
        self.perims = map(self.get_perim, range(len(self.faces)))
        
        self.energy = self.total_mechanical_energy()

    #I can't handle spherical coordinates so here
    def spherical_to_cartesian(self,sph):
        r,theta,phi = self.radius,sph[0],sph[1]
        x = r*np.sin(theta)*np.cos(phi)
        y = r*np.sin(theta)*np.sin(phi)
        z = r*np.cos(theta)

        return np.array([x,y,z])
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

    def make_angular_edge(self,idx1,idx2):
        #return the smallest-magnitude [theta,phi] pair that gets you from idx1 to idx2
        edge = make_edge(idx1,idx2)
        


    def get_edge(self,idx1,idx2):
        j = self.adj[idx1].index(idx2)
        if j == None:
            print 'edge doesn\'t exist'
            return
        return self.edges[idx1][j]
    def get_edges(self,idx):
        edgecopy = [self.edges[idx][i].copy() for i in range(3)]
        return edgecopy

    def find_face(self,v1,v2,v3):
        ## find the face that includes the three indices idx1-idx2-idx3
        edges2 = self.get_edges(v2)
        
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
            edg = self.get_edges(current_vertex)
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
