# Class implementation of the Lie group R2R+
import numpy as np
import torch

# Rules for setting up a group class:
# A group element is always stored as a 1D vector, even if the elements consist
# only of a scalar (in which case the element is a list of length 1). Here we 
# also assume that you can parameterize your group with a set of n parameters,
# with n the dimension of the group. The group elements are thus always lists of
# length n.
#
# This file requires the definition of the base/normal sub-group R^n and the 
# sub-group H. Together they will define G = R^n \rtimes H.
#
# In order to derive G (it's product and inverse) we need for the group H to be
# known the group product, inverse and left action on R^n.
#
# In the B-Spline networks we also need to know a distance between two elements
# g_1 and g_2 in H. This can be defined by computing the length (l2 norm) of the
# vector that brings your from element g_1 to g_2 via the exponential map. This 
# vector can be obtained by taking the logarithmic map of (g_1^{-1} g_2).
#
# Finally we need a way to sample the group. Therefore also a function "grid" is
# defined which samples the group as uniform as possible given a specified 
# number of elements N. Not all groups allow a uniform sampling given an 
# aribitrary N, in which case the sampling can be made approximately uniform by 
# maximizing the distance between all sampled elements in H (e.g. via a 
# repulsion model).


## The normal sub-group R^n:
# This is just the vector space R^n with the group product and inverse defined 
# via the + and -.
class Rn:
    # Label for the group
    name = 'R^2'
    # Dimension of the base manifold N=R^n
    n = 2 
    # The identity element
    e = torch.zeros(2)

## The sub-group H:
class H:
    # Label for the group
    name = 'R+'
    # Dimension of the sub-group H
    n = 1 # Each element consists of 1 parameter
    # The identify element
    e = torch.ones(1)

    ## Essential definitions of the group
    # Group product
    def prod(h_1, h_2):
        return h_1 * h_2

    # Group inverse
    def inv(h):
        return 1./h

    ## Essential for computing the distance between two group elements
    # Logarithmic map
    def log(h):
        return torch.log(h)

    def exp(v):
        return torch.exp(v)

    # Distance between two group elements
    def dist(h_1, h_2):
        # The logarithmic distance ||log(inv(h1)*h2)||
        dist = (H.log(H.prod(H.inv(h_1), h_2)))[...,0] # Since each h is a list of length 1 we can do [...,0]
        return torch.abs(dist)

    ## Essential for constructing the group G = R^n \rtimes H
    # Define how H acts transitively on R^n
    def left_action_on_Rn(h, xx):
        return xx

    ## Essential in the group convolutions
    # Define the determinant (of the matrix representation) of the group element
    def det(h):
        return h**(Rn.n)

    ## Essential for sampling of the group
    # Generating a grid
    def grid(N, dc): # dc: A list of which the length corresponds to the number of exponential coordinates (dimensionality of H)
        if N==0:
            h_list = torch.tensor([], dtype=torch.float32, requires_grad=False) 
        else:
            c_list = dc * torch.tensor(np.array([np.linspace(-(N-1)/2, (N-1)/2,N)]).T,
                    dtype=torch.float32, requires_grad=False)
            h_list = H.exp(c_list)
        return h_list

## For the dilation group we define the global and local grid in the same way
class grid_global: # For a global grid
    # Should a least contain:
    #	N     - specifies the number of grid points
    #	scale - specifies the (approximate) distance between points, this will be used to scale the B-splines
    # 	grid  - the actual grid
    #	args  - such that we always know how the grid was construted
    # Construct the grid
    def __init__(self, N, scale_range):
            # This rembembers the arguments used to construct the grid (this is to make it a bit more future proof, you may want to define a grid using specific parameters and later in the code construct a similar grid with the same parameters, but with N changed for example)
        self.args = locals().copy()
        self.args.pop('self')
        # Store N
        self.N = N
        # Define the scale (the spacing between points)
        # dc should be a list of length H.n. Let's turn it into a numpy array:
        if N > 1:
            scale_np = np.array(np.log(scale_range)/(N-1))
            self.scale = scale_np
        else:
            scale_np = np.array(1.)
            self.scale = scale_np

        # Generate the grid
        # Create an array of uniformly spaced exp. coordinates (step size is 1):
        # The indices always include 0. When N = odd, the grid is symmetric. E.g. N=3 -> [-1,0,1].
        # When N = even the grid is moved a bit to the right. E.g. N=2 -> [0,1], N=3 -> [-1,0,1,2]
        # grid_start = -((N-1)//2)
        c_index_array = np.moveaxis(np.mgrid[tuple([slice(0,N)]*H.n)],0,-1).astype(np.float32)
        # Scale the grid with dc
        c_array = scale_np*c_index_array
        # Flatten it to a list of exp coordinates as a tensorflow constant
        c_list = torch.tensor(np.reshape(c_array,[-1,H.n]), requires_grad=False)
        # Turn it into group elements via the exponential map
        h_list = H.exp(c_list)
        # Save the generated grid
        self.grid = h_list

	## For the dilation group we define the global and local grid in the same way
class grid_local: # For a global grid
        # Should a least contain:
        #	N     - specifies the number of grid points
        #	scale - specifies the (approximate) distance between points, this will be used to scale the B-splines
        # 	grid  - the actual grid
        #	args  - such that we always know how the grid was construted
        # Construct the grid
    def __init__(self, N, scale ):
            # This rembembers the arguments used to construct the grid (this is to make it a bit more future proof, you may want to define a grid using specific parameters and later in the code construct a similar grid with the same parameters, but with N changed for example)
        self.args = locals().copy()
        self.args.pop('self')
        # Store N
        self.N = N
        # Define the scale (the spacing between points)
        # dc should be a list of length H.n. Let's turn it into a numpy array:
        scale_np = np.array(scale)
        self.scale = scale_np

        # Generate the grid

        # Create an array of uniformly spaced exp. coordinates (step size is 1):
        # The indices always include 0. When N = odd, the grid is symmetric. E.g. N=3 -> [-1,0,1].
        # When N = even the grid is moved a bit to the right. E.g. N=2 -> [0,1], N=3 -> [-1,0,1,2]
        grid_start = -((N-1)//2)
        c_index_array = np.moveaxis(np.mgrid[tuple([slice(grid_start,grid_start + N)]*H.n)],0,-1).astype(np.float32)
        # Scale the grid with dc
        c_array = scale_np*c_index_array
        # Flatten it to a list of exp coordinates as a tensorflow constant
        c_list = torch.tensor(np.reshape(c_array,[-1,H.n]), requires_grad=False)
        # Turn it into group elements via the exponential map 
        h_list = H.exp(c_list)
        # Save the generated grid
        self.grid = h_list




## The derived group G = R^n \rtimes H.
# The above translation group and the defined group H together define the group G
# The following is automatically constructed and should not be changed unless
# you may have some speed improvements, or you may want to add some functions such 
# as the logarithmic and exponential map.
# A group element in G should always be a vector of length Rn.n + H.n
class G:
    # Label for the group G
    name = 'R2R+'
    # Dimension of the group G
    n = Rn.n + H.n
    # The identity element
    e = torch.cat([Rn.e,H.e])

    # Function for splitting a group element g in G in to its xx and h component
    def xx_h(g):
        xx = g[...,0:Rn.n]
        h = g[...,Rn.n:]
        return xx, h

    # Function that returns the classes for R^n and H
    def Rn_H():
        return Rn, H

    # The derived group product
    def prod(g_1, g_2): # Input g_1 is a single group element, input g_2 can be an array of group elements
        # (xx_1, h_1)
        xx_1 = g_1[0:Rn.n]
        h_1 = g_1[Rn.n:]
        # (xx_2, h_2)
        xx_2 = g_2[...,0:Rn.n]
        h_2 = g_2[...,Rn.n:]
        # (xx_new, h_new)
        xx_new = H.left_action_on_Rn(h_1, xx_2) + xx_1
        h_new = H.prod(h_1,h_2)
        g_new = torch.cat([xx_new,h_new],axis=-1)
        # Return the result
        return g_new

    # The derived group inverse
    def inv(g):
        # (xx, h)
        xx = g[0:Rn.n]
        h = g[Rn.n:]
        # Compute the inverse
        h_inv = H.inv(h)
        xx_inv = H.left_action_on_Rn(H.inv(h), -xx)
        g_inv = torch.cat([xx_inv,h_inv],axis=-1)
        return g_inv
