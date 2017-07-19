import numpy as np
import matplotlib.pyplot as plt
from vertex_model_2d import VertexModel
def same(x,y):
    return (x and y) or ((not x) and (not y))
def switchaforb(lis,a,b):
    lis[lis.index(a)] = b

class VertexModelApicoBasal:
    def __init__(self, locsb_in, adjs_in, k=1, h=2., A0=1./12, r=1, s=1, boxsizeb = 1.):
        #contains two vertex models, related by curvature.
        locsb = list(np.array(locsb_in).copy())
        adjs = [x[:] for x in adjs_in]
        self.basal_vm = VertexModel(locsb,adjs, boxsize=boxsizeb, A0=1)
        self.K0 = k #dimensionless curvature
        self.A0 = A0
        self.r = r
        self.s = s

        self.thickness = h #thickness has dimensions
        aratio = np.sqrt(1./np.power(1+k,2))
        boxsizea = aratio*boxsizeb
        self.apical_vm = VertexModel([i*aratio for i in locsb],adjs, boxsize=boxsizea, A0=1)

        self.lateralareas = [self.get_lateralarea(i) for i in range(len(self.basal_vm.faces))]

        self.energy = self.combined_energy()

    #dealing with the lateral surface areas:
    def slant_height_ba_vect(self,v1,v2):
        boxsizea = self.apical_vm.boxsize
        boxsizeb = self.basal_vm.boxsize
        
        apical_edge = self.apical_vm.get_edge(v1,v2)
        apical_edgelen = np.linalg.norm(apical_edge)
        apical_midpoint = self.apical_vm.get_vert(v1) + apical_edge/2.
    
        basal_edge = self.basal_vm.get_edge(v1,v2)
        basal_edgelen = np.linalg.norm(basal_edge)
        basal_midpoint = self.basal_vm.get_vert(v1) + basal_edge/2.


        vec_to_midpoint = np.array([boxsizeb/2., boxsizeb/2.]) - basal_midpoint
        apical_mp_trans = apical_midpoint+boxsizea/boxsizeb*vec_to_midpoint
        #put things inside the same periodic-boundary-conditions box
        if apical_mp_trans[0]>boxsizea:
            apical_mp_trans[0]-=boxsizea
        elif apical_mp_trans[0]<0:
            apical_mp_trans[0]+=boxsizea
        if apical_mp_trans[1]>boxsizea:
            apical_mp_trans[1]-=boxsizea
        elif apical_mp_trans[1]<0:
            apical_mp_trans[1]+=boxsizea
        apical_mp_trans = apical_mp_trans-np.array([boxsizea/2.,boxsizea/2.])
        return apical_mp_trans
    def average_edgelen(self,v1,v2):
        apical_edge = self.apical_vm.get_edge(v1,v2)
        apical_edgelen = np.linalg.norm(apical_edge)
    
        basal_edge = self.basal_vm.get_edge(v1,v2)
        basal_edgelen = np.linalg.norm(basal_edge)

        return (apical_edgelen + basal_edgelen)/2.

    def slant_height(self, v1, v2):
        #get the slant height of the lateral contact between vertices v1 and v2
        apical_mp_trans = self.slant_height_ba_vect(v1,v2)
        slant_height = np.sqrt(np.power(self.thickness,2) + np.sum(np.power(apical_mp_trans,2)))
        return slant_height

    def get_lateralarea(self, idx):
        #estimate the lateral surface area for the face with index idx
        face = self.apical_vm.faces[idx]
        boxsizea = self.apical_vm.boxsize
        boxsizeb = self.basal_vm.boxsize
        lateralarea = 0

        for v in range(len(face)):
            v1 = face[v-1]
            v2 = face[v]
            slant_height = self.slant_height(v1,v2)
            average_edgelen = self.average_edgelen(v1,v2)

            lateralarea += average_edgelen*slant_height

        return lateralarea
        #note: the apical and basal surfaces "line up" at the middle of the boxes.

    #mechanical energy
    def combined_energy(self):
        A0 = self.A0
        K0 = self.K0
        r = self.r
        s = self.s

        apical_faceareas = np.array(self.apical_vm.faceareas)
        basal_faceareas = np.array(self.basal_vm.faceareas)
        lateral_faceareas = np.array(self.lateralareas)

        A_tilde =  apical_faceareas + basal_faceareas + lateral_faceareas
        V_tilde = self.thickness*(apical_faceareas + basal_faceareas)/2
        
        K_tilde = np.sqrt(np.divide(apical_faceareas, basal_faceareas)) - 1

        return np.dot(V_tilde-1, V_tilde-1) + np.dot(A_tilde-A0, A_tilde-A0)/r + np.dot(K_tilde-K0, K_tilde-K0)/s

    #convenient terms in the energy gradient
    def del_slant_height_apical(self,v1,v2):
        slh = self.slant_height(v1,v2)
        shvec = self.slant_height_ba_vect(v1,v2)
        return 0.5/slh * shvec
    def del_slant_height_basal(self,v1,v2):
        slh = self.slant_height(v1,v2)
        shvec = self.slant_height_ba_vect(v1,v2)
        return -0.5/slh * shvec

    #gradient descent stuff
    def gradient_one_vertex(self, idx, apical):
        #energy gradient wrt moving one vertex. apical is a flag for whether you're moving the apical vertex.
        A0 = self.A0
        K0 = self.K0
        r = self.r
        s = self.s

        if apical:
            planar_vm = self.apical_vm
        else:
            planar_vm = self.basal_vm
        adj = planar_vm.adj[idx] 
        faceadj = planar_vm.faceadj[idx]
        faces =  map(lambda x: planar_vm.rotate_face(x,idx), faceadj)
        faces = map(lambda x: [faces[x][i] for i in range(len(faces[x])) if faces[x][i] in adj], range(len(faces)))

        gradient = 0
        for f in range(len(faceadj)):
            #compute the contribution to the gradient from each face.
            faceidx = faceadj[f]
            apical_facearea = self.apical_vm.faceareas[faceidx]
            basal_facearea = self.basal_vm.faceareas[faceidx]
            lateral_facearea = self.lateralareas[faceidx]

            A =  apical_facearea + basal_facearea + lateral_facearea
            V = self.thickness*(apical_facearea + basal_facearea)/2
            K = np.sqrt(np.divide(apical_facearea, basal_facearea)) - 1

            Aa = self.apical_vm.faceareas[faceidx]
            Ab = self.basal_vm.faceareas[faceidx]
            
            avg_len_1 = self.average_edgelen(faces[f][0],idx)
            avg_len_2 = self.average_edgelen(faces[f][1],idx)
            
            slh_1 = self.slant_height(faces[f][0],idx)
            slh_2 = self.slant_height(faces[f][1],idx)
            
            del_l1 = planar_vm.del_l(idx,faces[f][0])
            del_l2 = planar_vm.del_l(idx,faces[f][1])

            dA = planar_vm.del_A(idx,faceidx)
            if apical:
                dslh1 = self.del_slant_height_apical(faces[f][0],idx)
                dslh2 = self.del_slant_height_apical(faces[f][1],idx)
            else:
                dslh1 = self.del_slant_height_basal(faces[f][0],idx)
                dslh2 = self.del_slant_height_basal(faces[f][1],idx)

            del_V = self.thickness/2*dA
            del_A = dA + (dslh1*avg_len_1 + slh_1/2.*del_l1) + (dslh2*avg_len_2 + slh_2/2.*del_l2)
            if apical:
                del_K = -0.5*np.power(Ab/Aa,-0.5)*(Ab/np.power(Aa,2))*dA
            else:
                del_K = 0.5*np.power(Ab/Aa,-0.5)*(1/Aa)*dA

            gradient = gradient + 2*(V-1)*del_V + (2./r)*(A-A0)*del_A + (2./s)*(K-K0)*del_K

        return gradient
    def gradient_descent_one_vertex(self,idx, apical, gamma):
        # move a vertex against the energy gradient
        if apical:
            planar_vm = self.apical_vm
        else:
            planar_vm = self.basal_vm

        gradient = self.gradient_one_vertex(idx, apical)
        #print gradient
        
        #update position, edge lengths, face areas in the planar VertexModel
        newpos = planar_vm.v[idx]-gamma*gradient
        for i in range(2):
            if newpos[i]>planar_vm.boxsize:
                newpos[i] -= planar_vm.boxsize
            elif newpos[i]<0:
                newpos[i] += planar_vm.boxsize
        planar_vm.v[idx] = newpos
        planar_vm.edgelens[idx] = map(np.linalg.norm, planar_vm.get_edges(idx))
        for i in planar_vm.faceadj[idx]:
            planar_vm.faceareas[i] = planar_vm.get_facearea(i)
            planar_vm.perims[i] = planar_vm.get_perim(i)

            #update the lateral areas
            self.lateralareas[i] = self.get_lateralarea(i)

        E = self.energy
        self.energy = self.combined_energy()
        #print E-self.energy
        return self.energy-E

    def gradient_descent_step(self, gamma):
        step = 0
        for i in range(len(self.apical_vm.v)):
            step += self.gradient_descent_one_vertex(i, True, gamma)
            step += self.gradient_descent_one_vertex(i, False, gamma)
        return step
    
    def gradient_descent_no_t1(self, gamma, tol):
        step = 2*tol
        while abs(step)>tol:
            step = self.gradient_descent_step(gamma)
            if step>0:
                print 'ascending'
                
    #useful helper functions
    def show_model(self):
        plt.subplot(1,2,1)
        self.basal_vm.show_model()
        plt.title('Basal')
        plt.subplot(1,2,2)
        self.apical_vm.show_model()
        plt.title('Apical')
        plt.show()

    #t1 transition: must do it on both sides simultaneously.
    def t1_trans(self,idx1,idx2):
        (Aa,Ba,Ca,Da) = self.apical_vm.t1_trans(idx1,idx2)
        (Ab,Bb,Cb,Db) = self.basal_vm.t1_trans(idx1,idx2)

        if not ((Aa,Ba,Ca,Da)==(Ab,Bb,Cb,Db)):
            print 'error: apical and basal topologies do not agree in t1_trans.'
        
        self.lateralareas[Aa] = self.get_lateralarea(Aa)
        self.lateralareas[Ba] = self.get_lateralarea(Ba)
        self.lateralareas[Ca] = self.get_lateralarea(Ca)
        self.lateralareas[Da] = self.get_lateralarea(Da)

    def evaluate_t1_trans(self,i,k):
        #evaluate doing the t1 transition on both apical and basal sides
        params_a = self.apical_vm.cache_params()
        params_b = self.basal_vm.cache_params()

        E_notrans = self.combined_energy()

        self.t1_trans(i,k)

        E_withtrans = self.combined_energy()

        if E_withtrans>E_notrans:
            self.apical_vm.load_params(params_a)
            self.basal_vm.load_params(params_b)
            return 0
        self.energy = E_withtrans
        return 1
    
    def gradient_descent_with_t1(self, gamma, tol, l_crit):
        step = 2*tol
        total_trans = 0
        while abs(step)>tol:
            step = self.gradient_descent_step(gamma)
            for i in range(len(self.apical_vm.v)):
                for j in range(3):
                    if (self.apical_vm.edgelens[i][j]<l_crit) and (self.basal_vm.edgelens[i][j]<l_crit):
                        E_i = self.energy
                        k = self.apical_vm.adj[i][j]
                        total_trans += self.evaluate_t1_trans(i,k)
            if step>0:
                print 'ascending'
        print 'total T1 transitions:',total_trans

    def classify_rosettes(self,l_crit):
        apical_ros = self.apical_vm.find_rosettes(l_crit)
        basal_ros = self.basal_vm.find_rosettes(selfl_crit)

        shared_rosettes = []
        apical_rosettes = []

        for i in range(len(apical_ros)):
            shared = False
            bas_idx = None
            for j in range(len(basal_ros)):
                if apical_ros[i]==basal_ros[j]:
                    shared_rosettes += [apical_ros[i]]
                    bas_idx = j
                    shared = True
            if bas_idx!=None:
                basal_ros.pop(bas_idx)
            if not shared:
                apical_rosettes += [apical_ros[i]]
        basal_rosettes = basal_ros

        return apical_rosettes,basal_rosettes,shared_rosettes