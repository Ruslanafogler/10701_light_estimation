import skimage.io as ski_io
from skimage import color
import cv2
import numpy as np
from scipy.interpolate import interpn
from scipy.signal import convolve2d
import math
import matplotlib.pyplot as plt
from tqdm import tqdm

def read_img(path, do_downsample=False, n=200):
    print("imreading image", path)
    im = ski_io.imread(path)
    im = im.astype(np.double)
    #downsampling...
    if(do_downsample):
        im = im[::n,::n]  
    return im

def read_tiff(path, downsample=False, n=10):
    print("imreading image", path)
    im = ski_io.imread(path, plugin='tifffile')
    if(downsample):
        im = im[::n, ::n]
    print("img ranges min and max:", np.min(im), np.max(im))
    im = im.astype('uint8')
    return im


def gradient_per_channel(im_2d):

    #padding -> (top, bottom), (left, right)
    
    Ix = np.pad(im_2d, ((0,0), (1,0)), mode='constant')
    Iy = np.pad(im_2d, ((1,0), (0,0)), mode='constant')

    x_grad = np.diff(Ix[:, :], axis=1) #traverse L to R (along row)
    y_grad = np.diff(Iy[:, :], axis=0) #traverse up to down (along col)

    #return tuple of gradients, shave off added padding
    return x_grad[:, :], y_grad[:, :]


def divergence_per_channel(input_im):
    if(type(input_im) != tuple):
        gradx, grady = gradient_per_channel(input_im)
        #TODO idk if this is correct, technically
        div = gradx[:-1, :] + grady[:, :-1]
        return div
    else:
        input_gradx = input_im[0]
        input_grady = input_im[1]
        fxx = gradient_per_channel(input_gradx)[0]
        fyy = gradient_per_channel(input_grady)[1]

        #because of second derivative, things were shifted by 1
        #shift fxx by ROW to compensate (fyy reduction by col)
        #shift fyy by COLUMN to compensate (fxx reduction by row)
        #go up to -1 so that sizes are compatible (which just sacrifices border anyway)

        div = fxx[:-1, 1:] + fyy[1:, :-1]
        return div    
    
    


def laplacian_per_channel(im_2d):
    kernel = np.asarray([[0,  1, 0],
                           [1, -4, 1],
                           [0,  1, 0]])        
    convolved = convolve2d(im_2d, kernel, mode='same', boundary='fill', fillvalue=0)
    return convolved



def total_laplacian(I):

    laplace_transformed = np.zeros_like(I)
    for channel in range(3):
        convolved = laplacian_per_channel(I[:,:,channel])
        laplace_transformed[:,:,channel] = convolved
    return laplace_transformed


def confirm_laplacian():
    museum = read_img("data/museum/museum_ambient.png")
    #TODO: do I want alpha channel?
    museum = museum[:,:,:3]
    h,w,c = museum.shape
    print("museum shape", museum.shape)
    museum_R_channel = museum[:,:,0]
    #grad = gradient_per_channel(museum_R_channel)
    rhs = laplacian_per_channel(museum_R_channel)
    lhs = divergence_per_channel(gradient_per_channel(museum_R_channel))
    #print("top left corner of lhs\n", lhs[0:10, 0:10])
    print("shape of rhs", lhs.shape)
    print("shape of lapalcian", rhs.shape)
    print("top left corner of IMAGE \n", museum_R_channel[0:11, 0:11])    
    print("top left corner of laplacian\n", rhs[0:11, 0:11])
    print("top left corner of rhs \n", lhs[0:10, 0:10])

    print("top right corner of laplacian\n", rhs[-11:, 0:11])
    print("top right corner of rhs \n", lhs[-10:, 0:10])

    mask_interior = np.ones_like(rhs)
    mask_interior[0, :] = 0
    mask_interior[:,0] = 0
    mask_interior[-1, :] = 0
    mask_interior[:, -1] = 0
    print("mask is\n", mask_interior)
    print(lhs.shape, rhs.shape, mask_interior.shape)
    
    rhs[mask_interior == 0] = 0
    #pad lhs with 0s to act as zero'd out
    lhs = np.pad(lhs, ((0,1), (0,1)), mode='constant')
    uhoh = np.where(lhs != rhs)
    print("uhoh indices", uhoh)
    print("if uhoh indices are empty arrays, then yay! it worked")    

def inner(A, B):
    return np.sum(A*B)

def poisson_solve(D, I_init, B, I_boundary, eps, N):
    #initialization
    I = B * I_init + (1-B)*I_boundary
    r = B *(D - laplacian_per_channel(I)) #638x905
    d = r
    delta_new = inner(r,r)
    print("delta_new (dot of r.T r) shape", delta_new)
    n = 0

    print("I shape, r shape", I.shape, r.shape)

    pbar = tqdm(total = 1000)

    while(np.linalg.norm(delta_new) > eps and n < N):
        q = laplacian_per_channel(d) #q should be 638x905
        eta = delta_new/(inner(d,q)) # something / (905x638 times 638x905) -> something / 905x905
        I = I + B*(eta*d) #eta*d is 638x905, ideally, d is 638x905
        r = B*(r - eta*q)
        delta_old = delta_new
        delta_new = inner(r,r)
        Beta = delta_new/delta_old
        d = r + Beta * d
        n = n+1

        pbar.update(1)
        if(n > N):
            print("quitting, ran up to max iters")
    
    print("finished poisson solve")
    pbar.close()
    return I


def get_boundary_mask(I):
    B = np.ones_like(I)
    B[0, :] = 0
    B[:,0] = 0
    B[-1, :] = 0
    B[:, -1] = 0     
    return B  

def normalize_img(I):
    normalized = (I - np.min(I))/(np.max(I) - np.min(I))
    return np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)

def write_image(path, data):
    new_image = np.clip(data, 0.0, 1.0)
    saved = (new_image * 255).astype(np.uint8)
    ski_io.imsave(path, saved)


def mag_of_im(gradx, grady):
    return np.sqrt(np.square(gradx) + np.square(grady))

def grad_field(omega, M, grad_a_component, grad_phi_component):
    return omega*grad_a_component + (1-omega)*(M*grad_phi_component + (1-M)*grad_a_component)


def new_grad_field(ambient_im2d, flash_im2d, sigma=40, tau=0.9):
    
    grad_a_x, grad_a_y = gradient_per_channel(ambient_im2d)
    grad_phi_x, grad_phi_y = gradient_per_channel(flash_im2d)


    num = (grad_a_x * grad_phi_x + grad_a_y * grad_phi_y)
    den = mag_of_im(grad_a_x, grad_a_y) * mag_of_im(grad_phi_x, grad_phi_y)
    M = num/den
    M = np.nan_to_num(M, nan=1.0, posinf=1.0, neginf=1.0)


    omega = np.tanh(sigma * (flash_im2d - tau))
    omega = normalize_img(omega)
    
    new_grad_phi_x = grad_field(omega, M, grad_a_x, grad_phi_x)
    new_grad_phi_y = grad_field(omega, M, grad_a_y, grad_phi_y)

    return divergence_per_channel((new_grad_phi_x, new_grad_phi_y))



def grad_field_reflection(omega_ue, grad_a_component, grad_H_component):
    projectH_to_A = (grad_H_component * grad_a_component) / (np.sum(grad_a_component**2))
    return omega_ue * grad_H_component + (1.0 - omega_ue) * projectH_to_A


def new_grad_field_reflection(ambient_im2d, flash_im2d, sigma=0.1, tau_ue=0.1):
    grad_a_x, grad_a_y = gradient_per_channel(ambient_im2d)
    omega_ue = 1.0 - np.tanh(sigma * (flash_im2d - tau_ue))
    omega_ue = normalize_img(omega_ue)

    H = ambient_im2d + flash_im2d

    grad_H_x, grad_H_y = gradient_per_channel(H)
    
    new_grad_phi_x = grad_field_reflection(omega_ue, grad_a_x, grad_H_x)
    new_grad_phi_y = grad_field_reflection(omega_ue, grad_a_y, grad_H_y)

    return divergence_per_channel((new_grad_phi_x, new_grad_phi_y))




def test_poisson_solve(museum, B, eps, N):
    new_image = np.zeros_like(museum)
    for channel in range(3):
        im = museum[:,:,channel]
        div_channel = laplacian_per_channel(im)
        I_init = np.zeros_like(im)
        I_boundary = im
        I_boundary[B == 1] = 0
        result = poisson_solve(div_channel, I_init, B, I_boundary, eps, N)
        new_image[:,:,channel] = result    
    plt.imshow(new_image)
    plt.show()

'''
if boundary_condition = 0, then use ambient as boundary condition
if  boundary_condition = 1, then use flash as boundary condition
if  boundary_condition = 2, then use average of ambient/flash

note that init condition should not matter much
if init condition = 0, then use ambient image as init
if  init condition = 1, then use flash image as init
if  init condition = 2, then use all zeros as init
'''
def generate_fused_img(ambient, flash, boundary_condition, do_reflection=False, init_condition=0, sigma=40, tau=0.9):
    eps = 0.001
    N = 10000
    B = get_boundary_mask(ambient[:,:,0])
    I_boundary = np.zeros_like(ambient[:,:,0])

    new_image = np.zeros_like(ambient)
    for channel in range(3):
        channel_m = ambient[:,:,channel]        
        channel_m_flash = flash[:,:,channel]
        if(do_reflection):
            div_input = new_grad_field_reflection(channel_m, channel_m_flash,  sigma, tau)
        else:
            div_input = new_grad_field(channel_m, channel_m_flash,  sigma, tau)
        div_input = np.pad(div_input, ((0,1), (0,1)), mode='constant')
    

        if(init_condition == 0):
            I_init = channel_m
        elif(init_condition == 1):
            I_init = channel_m_flash
        else:
            I_init = np.zeros_like(channel_m)
        
        
        if(boundary_condition == 0):
            #this was aliasing and runing the og
            I_boundary = np.copy(channel_m)
        elif(boundary_condition == 1):
            #this was aliasing and runing the og
            I_boundary = np.copy(channel_m_flash)
        else:
            I_boundary = (channel_m_flash+channel_m)/2
        
         
            
        
        I_boundary[B == 1] = 0  
        print("solving poisson for channel", channel, "...")       
        result = poisson_solve(div_input, I_init, B, I_boundary, eps, N)           
        print("finished solving!")
        new_image[:,:,channel] = result

    return new_image


def load_museum_imgs():
    museum = cv2.imread('data/museum/museum_ambient.png', cv2.IMREAD_UNCHANGED)
    museum = cv2.cvtColor(museum, cv2.COLOR_BGRA2BGR)[:,:,::-1]

    museum_f = cv2.imread('data/museum/museum_flash.png', cv2.IMREAD_UNCHANGED)
    museum_f = cv2.cvtColor(museum_f, cv2.COLOR_BGRA2BGR)[:,:,::-1]
    museum = normalize_img(museum)
    museum_f = normalize_img(museum_f)
    h,w,c = museum.shape
    print("museum shape", museum.shape)
    print("range of image", np.min(museum), np.max(museum))

    return museum, museum_f

def load_reflection_imgs():
    reflection = cv2.imread('data/reflection/ambient.png', cv2.IMREAD_UNCHANGED)
    reflection = cv2.cvtColor(reflection, cv2.COLOR_BGRA2BGR)[:,:,::-1]

    reflection_f = cv2.imread('data/reflection/reflection.png', cv2.IMREAD_UNCHANGED)
    reflection_f = cv2.cvtColor(reflection_f, cv2.COLOR_BGRA2BGR)[:,:,::-1]
    reflection = normalize_img(reflection)
    reflection_f = normalize_img(reflection_f)
    h,w,c = reflection.shape
    print("reflection shape", reflection.shape)
    print("range of image", np.min(reflection), np.max(reflection))

    return reflection, reflection_f

def load_my_imgs(ambient_path, flash_path, downsample=False, n=100):
    ambient = read_img(ambient_path, downsample, n)
    flash = read_img(flash_path, downsample, n)
    
    ambient = normalize_img(ambient)
    flash = normalize_img(flash)
    return ambient, flash

def main():

    use_museum = True
    use_observation = False
    test_laplacian = False

    GET_ALL_PERMUTATIONS = False


    if(use_museum):
        sigma = 40
        tau = 0.9      
        ambient, flash = load_museum_imgs()
    elif(use_observation):   
        sigma = 0.15
        tau = 0.1            
        ambient, flash = load_reflection_imgs()
        ambient = ambient[0:610, :560]
        flash = flash[0:610, :560]
    else:
        sigma = 3
        tau = 0.2        
        ambient_path = "data/mine/v_ambient2.JPG"
        flash_path = "data/mine/v_flash.JPG"      
        #DOWNSAMPLING by 10 to speed this up for testing  
        ambient, flash = load_my_imgs(ambient_path, flash_path, True, 10)


    if(test_laplacian):
        confirm_laplacian()
        return    



    #TODO: this is terrible code but I'm horrifically lazy rn

    if(GET_ALL_PERMUTATIONS):
        sigmas = [40, 20, 10]
        taus = [0.9, 0.5, 0.2]
        boundary_choices = [0,1,2]
        init_choices = [0, 1, 2]

        for sigma in sigmas:
            for tau in  taus:
                for boundary_choice in boundary_choices:
                    for init_choice in init_choices:
                        new_image = generate_fused_img(ambient, flash, boundary_choice, init_condition=init_choice, sigma=sigma, tau=tau)
                        new_image = np.clip(new_image, 0.0, 1.0)
                        saved = (new_image * 255).astype(np.uint8)
                        name = f"museum_boundary{boundary_choice}_init{init_choice}_{sigma}_{tau}.jpg" if use_museum else "my_fused_result.jpg"
                        ski_io.imsave(name, saved)
    else:
        boundary_choice = 1 #ambient, flash, avg for 0-2 respectively
        init_choice = 2 #ambient, flash, all 0s for 0-2 respectively
        
        new_image = generate_fused_img(ambient, flash, boundary_choice, do_reflection=use_observation, init_condition=init_choice, sigma=sigma, tau=tau)    
        new_image = np.clip(new_image, 0.0, 1.0)
        saved = (new_image * 255).astype(np.uint8)
        name = f"museum_boundary{boundary_choice}_init{init_choice}_{sigma}_{tau}.jpg" if use_museum else "my_fused_result.jpg"
        ski_io.imsave(name, saved)        


    print("shape of new image", new_image.shape)


    f, axarr = plt.subplots(3,1, figsize=(30,20))
    axarr[0].imshow(ambient)
    axarr[0].set_title("Ambient Image")
    axarr[1].imshow(flash)
    axarr[1].set_title("flash image")
    axarr[2].imshow(new_image)
    axarr[2].set_title("Gradient Domain Fused Image")
    f.tight_layout()
    plt.show()


    return 1

if __name__ == "__main__":
    main()