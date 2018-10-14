import numpy as np   
import cv2
import cmath
import math
import scipy
from scipy import signal 
from skimage.measure import structural_similarity as ssim

from matplotlib import pyplot as plt

#for loops are written for making database of all 4 images and 7 kernels and calculating their PSNR and SSIM

for j in range(1,5):
    for i in range(1,8):
        kernel_image=cv2.imread('/home/hk/Documents/3rdsem/image_processing/assignments/assignment2/kernels/Kernel'+str(i)+'G_c.png',0)   
        blurry_image=cv2.imread('/home/hk/Documents/3rdsem/image_processing/assignments/assignment2/blurry_images/gt_'+str(j)+'_k_'+str(i)+'.png') 
        original_image=cv2.imread('/home/hk/Documents/3rdsem/image_processing/assignments/assignment2/Original_Images/gt_'+str(j)+'.jpg') 

        #kernel_image=kernel_image/np.sum(kernel_image)

        [b,g,r] = cv2.split(blurry_image)

        #[b,g,r] = cv2.split(original_image)
        height1= b.shape[0]
        width1= b.shape[1]
        M=width1
        N=height1

        width = kernel_image.shape[1]
        height = kernel_image.shape[0]
        kernel_image=np.vstack((kernel_image,np.zeros([height1-height,width])))
        kernel_image=np.hstack((kernel_image,np.zeros([height1,width1-width])))

        iota = 0+1j

        def resize_image(image,height_img,width_img):
    
            width = image.shape[1]
            height = image.shape[0]
            image=np.vstack((image,np.zeros([height_img-height,width])))
            image=np.hstack((image,np.zeros([height_img,width_img-width])))
    
            return image
        
        def psnr(img1, img2):
            #mse = numpy.mean( (img1 - img2) ** 2 )
            mse_calc_mean=(np.sum( (img1 - img2) ** 2 ))/(img1.shape[0]*img1.shape[1])
            #print('mse_1',mse)
            print('mse_calc_mean',mse_calc_mean)
            if mse_calc_mean == 0:
                return 100
            PIXEL_MAX = 255.0
            return 20 * math.log10(PIXEL_MAX / math.sqrt(mse_calc_mean))
        
        def ssim_1(img1,img2):
            ss=ssim(img1,img2,multichannel=True)
            return ss

        def func_fft(func):

            exponent_ux_M = np.exp(-1*iota*2*np.pi*np.array(np.fromfunction(lambda x, u: x*u/M, (M, M), dtype=complex)))
            exponent_vy_N = np.exp(-1*iota*2*np.pi*np.array(np.fromfunction(lambda y, v: y*v/N, (N, N), dtype=complex)))
            F_u_v =  np.array(np.matrix(exponent_ux_M)*np.matrix(func)*np.matrix(exponent_vy_N))
            return F_u_v
            

        def func_ifft(func):

            exponent_ux_M = np.exp(iota*2*np.pi*np.array(np.fromfunction(lambda x, u: x*u/M, (M, M), dtype=complex)))
            exponent_vy_N = np.exp(iota*2*np.pi*np.array(np.fromfunction(lambda y, v: y*v/N, (N, N), dtype=complex)))
            f_x_y =  np.array(np.matrix(exponent_ux_M)*np.matrix(func)*np.matrix(exponent_vy_N))/(M*N)
            return f_x_y
            

        def de_blurr(image,kernel_image):
            
            im_fft_image = func_fft(image)
            im_fft_kernel = func_fft(kernel_image) 
            im_ifft3_image = im_fft_image/im_fft_kernel
            restored_image = func_ifft(im_ifft3_image).real
            restored_image = (restored_image - restored_image.min())*255/(restored_image.max()-restored_image.min())
            
            return restored_image

        def blurr(image,kernel_image):
            
            im_fft_image = func_fft(image)
            im_fft_kernel = func_fft(kernel_image) 
            im_ifft3_image = im_fft_image*im_fft_kernel
            blurr_image = func_ifft(im_ifft3_image).real
            blurr_image = (blurr_image - blurr_image.min())*255/(blurr_image.max()-blurr_image.min())
            
            return blurr_image

        
        def LS_filter(image,kernal,gamma):
               
            h= image.shape[0]
            w= image.shape[1]

            kernal_fft = func_fft(kernal)
            image_fft = func_fft(image)
             
            laplace = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])
            laplace = resize_image(laplace,h,w)
            laplace_fft = func_fft(laplace)

            mag_laplace_fft_squared = np.conjugate(laplace_fft)*laplace_fft
            mag_kernal_fft_squared = np.conjugate(kernal_fft)*kernal_fft

            image_estimate_fft = (np.conjugate(kernal_fft)*image_fft)/(mag_kernal_fft_squared+gamma*mag_laplace_fft_squared)
            restored_image = func_ifft(image_estimate_fft).real    
            restored_image = (restored_image - restored_image.min())*255/(restored_image.max()-restored_image.min())

            return restored_image

        def weiner_filter_1(image,L,img2):

            fft_kernel=func_fft(img2)
            fft_image=func_fft(image)
            numerator=np.conjugate(fft_kernel)*fft_kernel
         
            denomenator=fft_kernel*(numerator+L)
            
            restored_image=(numerator/denomenator)*fft_image
            restored_image=func_ifft(restored_image)
            restored_image=restored_image.real
            restored_image = (restored_image - restored_image.min())*255/(restored_image.max()-restored_image.min())
            
            return restored_image    

        def write_into_text(psnr,ssim,file_name):
            file = open('/home/hk/Documents/3rdsem/image_processing/assignments/assignment2/restored_images_LS_metrics/'+file_name+'.txt','w') 
            
            file.write('psnr: ') 
            file.write(str(psnr))
             
            
            file.write('\nssim: ') 
            file.write(str(ssim)) 
            
            file.close() 

        def trunc_inv(blue,green,red,wn,kernel_image):

            b,a = scipy.signal.butter(10, wn, btype='low')
            blue=de_blurr(blue,kernel_image)
            green=de_blurr(green,kernel_image)
            red=de_blurr(red,kernel_image)
            
            blue=scipy.signal.filtfilt(b, a, blue)
            blue=scipy.signal.filtfilt(b, a, blue.T)
            
            green=scipy.signal.filtfilt(b, a, green)
            green=scipy.signal.filtfilt(b, a, green.T)
            
            red=scipy.signal.filtfilt(b, a, red)
            red=scipy.signal.filtfilt(b, a, red.T)
            
            blue = (blue - blue.min())*255/(blue.max()-blue.min())
            red = (red - red.min())*255/(red.max()-red.min())
            green = (green - green.min())*255/(green.max()-green.min())
            restored_image = np.dstack((blue.T,green.T,red.T)).astype(np.uint8)
            
            return restored_image


        #**This is line for inverse filter
        #restored_image = np.dstack((de_blurr(b,kernel_image),de_blurr(g,kernel_image),de_blurr(r,kernel_image))).astype(np.uint8)

        #**These lines are for weiner filter
        #L=float(input("Value of filter_coeff? "))
        ##L=100000
        #restored_image = np.dstack((weiner_filter_1(b,L,kernel_image),weiner_filter_1(g,L,kernel_image),weiner_filter_1(r,L,kernel_image))).astype(np.uint8)

        #**These lines are for Truncated Inverse filter
        #wn =float(input("Enter wn:  "))
        ##wn=0.96
        #restored_image = trunc_inv(b,g,r,wn,kernel_image)

        #**These lines are for Least Squares filter
        # gamma =float(input("Enter ls_coeff:\t"))
        gamma=10e-3
        b = LS_filter(b,kernel_image,gamma)
        g = LS_filter(g,kernel_image,gamma)
        r = LS_filter(r,kernel_image,gamma)
        restored_image = np.dstack((b,g,r)).astype(np.uint8)

        # kernel_image=cv2.imread('/home/hk/Documents/3rdsem/image_processing/assignments/assignment2/kernels_crop/Kernel5G_c.png',0)   
        # kernel_image=kernel_image/np.sum(kernel_image)
        # original_image=mpimg.imread('/home/hk/Documents/3rdsem/image_processing/assignments/assignment2/Original_Images/gt_1.jpg')    
        # blurr_image=mpimg.imread('/home/hk/Documents/3rdsem/image_processing/assignments/assignment2/blurr_image_5.jpg')  
        # [r,g,b] = cv2.split(blurr_image)



        # plt.subplot(1,2,1),plt.imshow(restored_image)
        # plt.subplot(1,2,2),plt.imshow(image)
        # plt.show()


        #restored_image = np.dstack((blurr(b),blurr(g),blurr(r))).astype(np.uint8)
        write_into_text(psnr(original_image,restored_image),ssim_1(original_image,restored_image),'gt_'+str(j)+'_k_'+str(i))
        cv2.imwrite('/home/hk/Documents/3rdsem/image_processing/assignments/assignment2/restored_images_LS_metrics/gt_'+str(j)+'_k_'+str(i)+'.png',restored_image)

        #cv2.imshow('output_image',restored_image)
        #cv2.waitKey()

