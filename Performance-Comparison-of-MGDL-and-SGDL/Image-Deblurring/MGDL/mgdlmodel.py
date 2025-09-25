import jax.numpy as jnp
import numpy as np
from jax import jit, grad, random
from jax.example_libraries import stax, optimizers
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import os, imageio
import sys
from memory_profiler import profile
from skimage.metrics import peak_signal_noise_ratio as psnr
from jax import lax



#set up data
def data_setup(opt):

    key = random.PRNGKey(0)

    if opt.image == 'butterfly':
        image_path = 'image/butterfly.gif'  
        img = imageio.imread(image_path)/ 255.
    elif opt.image == 'male':
        image_path = 'image/male.tiff'  
        img = imageio.imread(image_path)/ 255. 
    elif opt.image == 'chest':
        image_path = 'image/chest.png'  
        img = imageio.imread(image_path)/ 255. 
    
    plt.imshow(jnp.uint8(255*img), cmap='gray')
    plt.axis('off')
    plt.title(f"Clean image", fontsize=20)
    fig_filename = f'Fig/MultiGrade_deblurring_deepbutterfly_CleanImage.png'
    plt.savefig(fig_filename, format='png', bbox_inches='tight', pad_inches=0.1)
    plt.show()    
    
    # Create jnput pixel coordinates in the unit square
    coords_x = jnp.linspace(0, 1, img.shape[0], endpoint=False)
    coords_y = jnp.linspace(0, 1, img.shape[1], endpoint=False)
    train_x = jnp.stack(jnp.meshgrid(coords_y, coords_x), -1)

    noise = opt.noise_level * random.normal(key, img.shape)
    train_y = apply_gaussian_blur(img, opt) + noise


    plt.imshow(train_y, cmap='gray')
    plt.title('Noise image')
    plt.show()

    plt.imshow(jnp.uint8(255*train_y), cmap='gray')
    plt.axis('off')
    plt.title(f"Blurring image: $\sigma = {opt.sigma}$", fontsize=20)
    fig_filename = f'Fig/MultiGrade_deblurring{opt.sigma}_deepbutterfly_BlurredImage.png'
    plt.savefig(fig_filename, format='png', bbox_inches='tight', pad_inches=0.1)
    plt.show()  

    noisyPSNR = psnr(img, train_y)

    print(f'noisy PSNR {noisyPSNR}')


    data = {}
    data['train_x'] = train_x
    data['train_y'] = train_y
    data['img'] = img


    return data


def gaussian_blur_kernel(opt):
    """
    Create a 2D Gaussian blur kernel using JAX.
    
    Parameters:
    - size: Size of the Gaussian kernel (must be odd).
    - sigma: Standard deviation of the Gaussian distribution.
    
    Returns:
    - kernel: Normalized 2D Gaussian kernel.
    """
    # Create a coordinate grid
    ax = jnp.arange(-opt.size // 2 + 1, opt.size // 2 + 1)
    xx, yy = jnp.meshgrid(ax, ax)
    
    # Compute Gaussian kernel
    kernel = jnp.exp(-(xx**2 + yy**2) / (2 * opt.sigma**2))
    
    # Normalize the kernel to ensure sum equals 1
    kernel /= jnp.sum(kernel)
    print(f"Kernel size is: {jnp.shape(kernel)}")
    return kernel

def apply_gaussian_blur(image, opt):
    """
    Apply Gaussian blur to an image using a 2D Gaussian kernel with symmetric padding.

    Parameters:
    - image: Input image (grayscale or color).
    - opt: Options containing `size` and `sigma` for Gaussian kernel.

    Returns:
    - blurred: Blurred image.
    """
    # Generate the Gaussian kernel
    kernel = gaussian_blur_kernel(opt)

    # Compute padding sizes based on the kernel size
    pad_h = (kernel.shape[0] - 1) // 2
    pad_w = (kernel.shape[1] - 1) // 2

    # Reshape the kernel to have the correct shape for convolution
    kernel = kernel[..., jnp.newaxis, jnp.newaxis]  # Shape: (height, width, 1, 1)

    # Convert image to JAX array and add channel dimension if necessary
    image = jnp.array(image)

    image = image[..., jnp.newaxis]  # Add a singleton channel dimension -> (height, width, 1)
    # Add batch dimension for convolution: Shape -> (1, height, width, 1)
    image = image[jnp.newaxis, ...]

    # Apply symmetric padding (reflects the boundary values)
    image_padded = jnp.pad(
        image,
        ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)),  # Pad height and width dimensions
        mode='symmetric'
    )
    
    # Apply convolution without additional padding (padding already handled)
    blurred = lax.conv_general_dilated(
        image_padded,  # Padded input image
        kernel,
        window_strides=(1, 1),  # Stride of 1 in both dimensions
        padding='VALID',  # No additional padding needed
        dimension_numbers=('NHWC', 'HWIO', 'NHWC')  # Standard dimension format
    )

    return blurred.squeeze()  # Shape: (height, width) for grayscale


def snn(grade, scaleFactor, accumulation_img, opt):
    
    # Define the layers for the x ijnput
    layers_x = []
    for i in range(opt.num_layer):
        layers_x.append(stax.Dense(opt.num_channel))
        
        if opt.activation == 'tanh':
            layers_x.append(stax.elementwise(jnp.tanh))
        elif opt.activation == 'relu':
            layers_x.append(stax.Relu)

    # Final dense layer for x
    _, snn_feature = stax.serial(*layers_x)
    
    layers_x.append(stax.Dense(1))  # Output layer
    
    # Define a custom layer to accumulate the image with the output
    def accumulate_output_layer(x):
        return scaleFactor * jnp.squeeze(x) + accumulation_img
        
    # Adding the custom accumulate_output_layer
    layers_x.append(stax.elementwise(accumulate_output_layer))
    
    # Create the model using stax.serial
    init_fn, model_fn = stax.serial(*layers_x)

    return init_fn, model_fn, snn_feature


@jit
def tv_matrix_form(pred_imgs):
    """Optimized matrix form of TV: compute the total variation."""
    # Compute the gradients (forward differences) for TV regularization
    TV_x = jnp.diff(pred_imgs, axis=0, append=pred_imgs[-1:, :])  # Vertical differences
    TV_y = jnp.diff(pred_imgs, axis=1, append=pred_imgs[:, -1:])  # Horizontal differences

    # Stack them to create the TV matrix
    TV = jnp.vstack([TV_x, TV_y])
    return TV

@jit
def prox_l1_u(u, alpha, beta, lambd, BN_Theta):
    """Compute prox_{alpha lambd / beta || ||_1}(alpha * BN_Theta + (1 - alpha) * u)"""
    temp_u = alpha * BN_Theta + (1 - alpha) * u
    return jnp.sign(temp_u) * jnp.maximum(jnp.abs(temp_u) - alpha * lambd / beta, 0)

# Train model with given hyperparameters and data
def train_model(grade, data, scaleFactor, train_data, accumulation_img, u, opt):

    #-----------------------------------support function-----------------------------------------
    rand_key = random.PRNGKey(0)
    # Model and loss functions
    init_fn, model_fn, snn_feature = snn(grade, scaleFactor, accumulation_img, opt)

    model_pred = jit(lambda params, inputs_x : model_fn(params, inputs_x))   
    snn_feature_pred = jit(lambda params, inputs_x : snn_feature(params, inputs_x))
    model_loss = jit(lambda params, u, inputs_x, output: 
                     .5 * jnp.sum((apply_gaussian_blur(model_pred(params, inputs_x), opt) - output) ** 2) +  
                     .5 * opt.beta * jnp.sum((tv_matrix_form(model_pred(params, inputs_x)) - u)**2) + opt.lambd * jnp.sum(jnp.abs(u))
                     )
    
    mseloss = jit(lambda params, inputs_x, output: jnp.sqrt(jnp.mean((model_pred(params, inputs_x) - output) ** 2) ))
    
    
    model_psnr = jit(lambda pred_img, org_img: -10 * jnp.log10(jnp.mean((pred_img - org_img) ** 2)))
    model_grad_loss = jit(lambda params, u, inputs_x, output: grad(model_loss)(params, u, inputs_x, output))
    
                
    opt_init, opt_update, get_params = optimizers.adam(opt.lr_params)
    opt_update = jit(opt_update)
    _, params = init_fn(rand_key, (-1, train_data[0].shape[-1]))
    opt_state = opt_init(params)

    psnrs = []
    train_losses = []
    xs = []

    for i in tqdm(range(opt.epoch)):        
        BN_Theta = tv_matrix_form(model_pred(params, train_data[0]))
        #update u
        u = prox_l1_u(u, opt.alpha, opt.beta, opt.lambd, BN_Theta)
        #update params
        opt_state = opt_update(i, model_grad_loss(params, u, *train_data), opt_state)
        params = get_params(opt_state)

        if i % opt.interval == 0:
            pred_img = model_pred(params, train_data[0])
            train_losses.append(model_loss(params, u, *train_data))
            psnrs.append(model_psnr(pred_img, data["img"]))
            xs.append(i)
            
    # confirm the last one is recorded
    pred_img = model_pred(params, train_data[0])
    train_losses.append(model_loss(params, u, *train_data))
    psnrs.append(model_psnr(pred_img, data["img"]))
    xs.append(i)

    train_features = snn_feature_pred(params, train_data[0])  
    NewscaleFactor = mseloss(params,  *train_data)

    return {
        'psnrs': psnrs,
        'accumulation_img': pred_img,
        'train_features': train_features,
        'train_losses': train_losses,
        'NewscaleFactor': NewscaleFactor,
        'xs': xs,
        'u': u
    }


def MGDLmodel(opt, data):

    
    train_features = data["train_x"]
    train_y = data["train_y"]
    accumulation_img = jnp.zeros_like(data['img'])
    u = jnp.zeros_like(tv_matrix_form(data['img']))

    scaleFactor = 1

    SaveHistory = {}


    for grade in range(1, opt.grade+1):
        
        train_data = [train_features, train_y]
        
        s_time = time.time() 
        history = train_model(grade, data, scaleFactor, train_data, accumulation_img, u, opt)
        e_time = time.time()

        train_features = history['train_features']
        u = history['u']
        scaleFactor = history['NewscaleFactor']
        accumulation_img = history['accumulation_img']
        
    

        SaveHistory['grade'+str(grade)] = {
            'psnrs': history['psnrs'],
            'train_losses': history['train_losses'],
            'accumulation_img': history['accumulation_img'],
            'xs': history['xs'],
            'scaleFactor': history['NewscaleFactor'],
            'time': e_time - s_time
        }

    picklename = 'results/MGDL_grade%d_lrparams%.2e_beta%.2e_lambd%.2e_psnr%.4e_loss%.4e.pickle' % (
        opt.grade, opt.lr_params, opt.beta, opt.lambd, history['psnrs'][-1], history['train_losses'][-1]
    )

    
    with open(picklename, 'wb') as f:
        pickle.dump([SaveHistory, opt], f)       



# @profile
def analysis(filepath, LossPsnr_print, Fig_print, grade):

    with open(filepath, 'rb') as f:
        [SaveHistory, opt] = pickle.load(f)


    print(f'noise level: {opt.noise_level}')

    print(SaveHistory.keys())

    data = data_setup(opt)

    

    current_epoch = 0
    MUL_EPOCH = [0]
    MUL_PSNR = []
    MUL_LOSS = []
    total_time = 0
    
    for grade in range(1, grade+1):

        current_epoch += opt.epoch
        MUL_EPOCH.append(current_epoch)
        
        history_dic = SaveHistory['grade'+str(grade)]
        psnrs = history_dic['psnrs'][1:]
        train_losses = history_dic['train_losses'][1:]
        pred_imgs = history_dic['accumulation_img']
        time = history_dic['time']
        MUL_PSNR.extend(psnrs)
        MUL_LOSS.extend(train_losses)
        
        total_time += time

        print(f'at grade {grade}, PSNR is {psnrs[-1]}, train_losses is {train_losses[-1]}, scaleFactor is {history_dic["scaleFactor"]}, time: {time}')

        if Fig_print:
            plt.imshow(jnp.uint8(255*pred_imgs), cmap='gray')
            plt.axis('off')
            plt.title(f"Multi-Grade: Grade {grade}", fontsize=20)
            fig_filename = f'Fig/MultiGrade_deblurring{opt.sigma}_deepbutterfly_predGrade{grade}.png'
            plt.savefig(fig_filename, format='png', bbox_inches='tight', pad_inches=0.1)
            plt.show()       
        

            

    print(f'the total time is {total_time}')


    if LossPsnr_print:

        epochs = [opt.interval * (i+1) for i in range(len(MUL_PSNR))]
    
        # plt.figure(figsize=(6.4,4.8))
        # Plot training and validation PSNR
        plt.plot(epochs, MUL_PSNR, label='PSNR')
        plt.title('Multi-Grade', fontsize=20)
        plt.xlabel('Epochs', fontsize=20)
        plt.ylabel('PSNR', fontsize=20)
        # plt.legend(fontsize=20)
        plt.ylim([18, 27.5])
        for x in MUL_EPOCH:
            plt.axvline(x, color='k', linestyle=':')
        plt.xticks(MUL_EPOCH) 
        plt.yticks(fontsize=20) 
        plt.xticks(fontsize=20)
        plt.tight_layout()

        fig_filename = f'Fig/MultiGrade_deblurring{opt.sigma}_deepbutterfly_PSNR.png'
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        plt.savefig(fig_filename, format='png', bbox_inches='tight', pad_inches=0.1)

        plt.show()
    
        # plt.figure(figsize=(6.4,4.8))
        # Plot training and validation PSNR
        plt.plot(epochs[1:], MUL_LOSS[1:], label='Loss')
        plt.title('Multi-Grade', fontsize=20)
        plt.xlabel('Epochs', fontsize=20)
        plt.ylabel('Loss', fontsize=20)
        # plt.legend(fontsize=20)
        for x in MUL_EPOCH:
            plt.axvline(x, color='k', linestyle=':')
        plt.xticks(MUL_EPOCH) 
        plt.yticks(fontsize=20) 
        plt.xticks(fontsize=20)
        plt.tight_layout()
        fig_filename = f'Fig/MultiGrade_deblurring{opt.sigma}_deepbutterfly_Loss.png'
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        plt.savefig(fig_filename, format='png', bbox_inches='tight', pad_inches=0.1)
        plt.show()


