import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
from numpy import pi
from mpl_toolkits.mplot3d import Axes3D
import mayavi.mlab as mlab
import pickle
from vertex_model_2d_sphere import VertexModelSphere

class VertexModelSphereAB:
    def __init__(self,locs, adjs, r=1.,a0=1.,radius=1.,h=0.5):
        #radius is inner radius, h is thickness.
        self.r = r
        self.a0=a0

        self.radius = radius
        self.thickness = h

        self.apical_vm = VertexModelSphere(locs,adjs,radius=radius)
        self.basal_vm = VertexModelSphere(locs,adjs,radius=(radius+h))
        self.lateralareas = map(self.make_lateral_area,range(len(self.apical_vm.faces)))
        self.V0 = 4./3*pi*(np.power(radius+h,3)-np.power(radius,3))/len(self.apical_vm.faces)


        self.energy = self.total_mechanical_energy()
        
    # geometry
    def average_edgelen(self,v1,v2):
        apical_edgelen = self.apical_vm.get_edgelen(v1,v2)
    
        basal_edgelen = self.basal_vm.get_edgelen(v1,v2)

        return (apical_edgelen + basal_edgelen)/2.
    def make_slant_height(self,idx1,idx2):
        apical_edge = self.apical_vm.get_edge(idx1,idx2)
        apical_mp = self.apical_vm.spherical_to_cartesian(self.apical_vm.cartesian_to_spherical(self.apical_vm.get_vert_cart(idx1) + apical_edge/2.))
        
        basal_edge = self.basal_vm.get_edge(idx1,idx2)
        basal_mp = self.basal_vm.spherical_to_cartesian(self.basal_vm.cartesian_to_spherical(self.basal_vm.get_vert_cart(idx1) + basal_edge/2.))

        return norm(apical_mp-basal_mp)
    def make_lateral_area(self,fidx):
        face = self.apical_vm.get_face(fidx)
        lateralarea = 0

        for v in range(len(face)):
            v1 = face[v-1]
            v2 = face[v]
            slant_height = self.make_slant_height(v1,v2)
            average_edgelen = self.average_edgelen(v1,v2)

            lateralarea += average_edgelen*slant_height

        return lateralarea


    # energy calculations
    def total_mechanical_energy(self):
        a0 = self.a0
        r = self.r
        V0 = self.V0

        apical_faceareas = np.array(self.apical_vm.faceareas)
        basal_faceareas = np.array(self.basal_vm.faceareas)
        lateral_faceareas = np.array(self.lateralareas)

        A_tilde =  (apical_faceareas + basal_faceareas + lateral_faceareas)/(np.power(V0,2./3))
        V_tilde = (self.thickness*(apical_faceareas + basal_faceareas)/2)/V0


        return np.dot(V_tilde-1, V_tilde-1) + np.dot(A_tilde-a0, A_tilde-a0)/r
    def mech_energy_vertex(self,idx):
        a0 = self.a0
        r = self.r
        V0 = self.V0

        faces = self.apical_vm.faceadj[idx]

        apical_faceareas = np.array([self.apical_vm.faceareas[f] for f in faces])
        basal_faceareas = np.array([self.basal_vm.faceareas[f] for f in faces])
        lateral_faceareas = np.array([self.lateralareas[f] for f in faces])

        A_tilde =  (apical_faceareas + basal_faceareas + lateral_faceareas)/(np.power(V0,2./3))
        V_tilde = (self.thickness*(apical_faceareas + basal_faceareas)/2)/V0


        return np.dot(V_tilde-1, V_tilde-1) + np.dot(A_tilde-a0, A_tilde-a0)/r

    # moving vertices
    def move_vertex(self,idx,new_pos,apical):
        if apical:
            polar_vm = self.apical_vm
        else:
            polar_vm = self.basal_vm

        polar_vm.move_vertex(idx,new_pos)
        faces = polar_vm.faceadj[idx]
        for f in faces:
            self.lateralareas[f] = self.make_lateral_area(f)
    def move_vertex_by(self,idx,increment,apical):
        if apical:
            polar_vm = self.apical_vm
        else:
            polar_vm = self.basal_vm

        polar_vm.move_vertex_by(idx,increment)
        faces = polar_vm.faceadj[idx]
        for f in faces:
            self.lateralareas[f] = self.make_lateral_area(f)
    
    #Gradient descent
    def gradient_descent_one_vertex(self,idx,thetastep,phistep):
        
        #Apical
        best_theta = 0
        best_phi = 0
        bestE = self.mech_energy_vertex(idx)
        for dtheta in [-thetastep,0,thetastep]:
            for dphi in [-phistep,0,phistep]:
                increment = np.array([dtheta,dphi])
                self.move_vertex_by(idx,increment,True)
                E = self.mech_energy_vertex(idx)
                if E<bestE:
                    bestE = E
                    best_theta = dtheta
                    best_phi = dphi
                self.move_vertex_by(idx,-increment,True)
        self.move_vertex_by(idx,np.array([best_theta,best_phi]),True)

        #Basal
        best_theta = 0
        best_phi = 0
        bestE = self.mech_energy_vertex(idx)
        for dtheta in [-thetastep,0,thetastep]:
            for dphi in [-phistep,0,phistep]:
                increment = np.array([dtheta,dphi])
                self.move_vertex_by(idx,increment,False)
                E = self.mech_energy_vertex(idx)
                if E<bestE:
                    bestE = E
                    best_theta = dtheta
                    best_phi = dphi
                self.move_vertex_by(idx,-increment,False)
        self.move_vertex_by(idx,np.array([best_theta,best_phi]),False)
    def gradient_descent_step(self,thetastep,phistep):
        energy_before = self.total_mechanical_energy()
        for idx in range(len(self.apical_vm.v)):
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
    
    # drawing things
    def draw_model(self):
        fig = mlab.figure()
        self.apical_vm.draw_model(fig,color=(1,0,0))
        self.basal_vm.draw_model(fig,color=(0,0,1))
        for i in range(len(self.apical_vm.v)):
            pt1 = self.apical_vm.v_cart[i]
            pt2 = self.basal_vm.v_cart[i]
            mlab.plot3d([pt1[0],pt2[0]],
                    [pt1[1],pt2[1]],
                    [pt1[2],pt2[2]],figure=fig,tube_radius=None,color=(0,1,0))
        mlab.show()

    # measuring geometric things
    def edge_constriction_histogram(self):
        '''
        How does each edge constrict basally to apically? 
        naively you expect a ratio of (inner radius)/(outer radius), 
        but different parameters would likely change that
        '''
        apical_edges = np.array(self.apical_vm.edgelens)
        basal_edges = np.array(self.basal_vm.edgelens)
        ratio = np.divide(apical_edges,basal_edges)

        hist, bins = np.histogram(ratio,bins='auto')
        width = 0.7 * (bins[1] - bins[0])
        center = (bins[:-1] + bins[1:]) / 2
        plt.bar(center, hist, align='center', width=width)
        plt.plot([self.radius/(self.radius+self.thickness), self.radius/(self.radius+self.thickness)], [0,max(hist)])
        plt.show()
        return ratio
    def count_rosettes(self,l_crit):
        apical_rosettes,apical_degrees = self.apical_vm.analyze_rosettes(l_crit, showplot=False)
        basal_rosettes,basal_degrees = self.basal_vm.analyze_rosettes(l_crit, showplot=False)

        apical_only = [i for i in apical_rosettes if i not in basal_rosettes]
        basal_only = [i for i in basal_rosettes if i not in apical_rosettes]
        return {'apical':len(apical_rosettes), 'basal':len(basal_rosettes), 'apical only': len(apical_only), 'basal only':len(basal_only)}
    # saving things
    def save_to_file(self, filename):
        #pickle v, adjs, radius, p0, and r.
        #importantly, we're not pickling the actual VertexModel instance so that I can change object methods without messing everything up

        with open(filename,'wb') as pickle_file:
            data = {'apical': {'locs':self.apical_vm.v, 'adjs':self.apical_vm.adj, 'radius':self.apical_vm.radius},
            'basal':{'locs':self.basal_vm.v, 'adjs':self.basal_vm.adj, 'radius':self.basal_vm.radius},
            'radius':self.radius,
            'thickness':self.thickness,
            'lateralareas':self.lateralareas,
            'V0':self.V0,
            'r':self.r,
            'a0':self.a0}
            pickle.dump(data, pickle_file)
    @staticmethod
    def load_from_file(filename):
        with open(filename,'rb') as pickle_file:
            data = pickle.load(pickle_file)
        vm = VertexModelSphereAB(data['basal']['locs'],data['basal']['adjs'],radius=data['radius'],a0=data['a0'], r=data['r'], h=data['thickness'])
        vm.apical_vm = VertexModelSphere(data['apical']['locs'], data['apical']['adjs'],radius=data['radius'])
        vm.lateralareas = data['lateralareas']
        vm.V0 = data['V0']
        vm.energy = vm.total_mechanical_energy()
        return vm
