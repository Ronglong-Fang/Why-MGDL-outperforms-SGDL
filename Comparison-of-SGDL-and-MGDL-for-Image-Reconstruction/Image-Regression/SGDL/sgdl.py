import numpy
import jax
import jax.numpy as np
from jax import jit, grad, random
from jax.example_libraries import stax, optimizers
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import os, imageio
from jax.scipy.signal import convolve
from PIL import Image

#set up data
def data_setup(opt):

    if opt.image == 'cameraman':
        image_path = 'image/cameraman.tif'  
        img = imageio.imread(image_path)/ 255.
    elif opt.image == 'barbara':
        image_path = 'image/barbara.gif'  
        img = imageio.imread(image_path)/ 255.
    elif opt.image == 'butterfly':
        image_path = 'image/butterfly.gif'  
        img = imageio.imread(image_path)/ 255.    
    elif opt.image == 'male':
        image_path = 'image/male.tiff'  
        img = imageio.imread(image_path)/ 255.
    elif opt.image == 'chest':
        image_path = 'image/chest.png'  
        img = imageio.imread(image_path)/ 255. 
    elif opt.image == 'walnut':
        image_path = 'image/walnut.png'  
        img = Image.open(image_path)
        # Convert the image to grayscale
        img = img.convert('L')
        img = np.array(img)/ 255.

    print(f'the shape of image is: {np.shape(img)}')
    
    plt.imshow(img, cmap='gray')
    plt.show()
    
    # Create input pixel coordinates in the unit square
    coords_x = np.linspace(0, 1, img.shape[0], endpoint=False)
    coords_y = np.linspace(0, 1, img.shape[1], endpoint=False)

    
    test_X = np.stack(np.meshgrid(coords_y, coords_x), -1)
    test_Y = img
    train_X = test_X[::2, ::2]
    train_Y = test_Y[::2, ::2]
    val_X = test_X[1::2, 1::2]
    val_Y = test_Y[1::2, 1::2]


    data = {} 
    
    data['train_X'] = train_X
    data['train_Y'] = train_Y
    data['val_X'] = val_X
    data['val_Y'] = val_Y
    data['test_X'] = test_X
    data['test_Y'] = test_Y

    print(f"shape of train_X: {np.shape(train_X)}, shape of train_Y: {np.shape(train_Y)}")
    print(f"shape of test_X: {np.shape(test_X)}, shape of test_Y: {np.shape(test_Y)}")



    return data, opt



# JAX network definition
def create_network(opt):

    layers = []

    for i in range(len(opt.num_channel)-2):
        layers.append(stax.Dense(opt.num_channel[i+1]))
        layers.append(stax.Relu)

    layers.append(stax.Dense(opt.num_channel[-1]))
    layers.append(stax.Identity)
        
    return stax.serial(*layers)




# Train model with given hyperparameters and data
def train_model(opt, train_data, val_data, test_data):

    key = random.PRNGKey(0)
    
    s_time = time.time() 
    init_fn, model_fn = create_network(opt)
    model_pred = jit(lambda params, x : np.squeeze(model_fn(params, x)))   
    model_loss = jit(lambda params, x, y: .5 * np.mean((model_pred(params, x) - y) ** 2))
    model_grad_loss = jit(lambda params, x, y: grad(model_loss)(params, x, y))
    model_psnr = jit(lambda loss: -10 * np.log10(2.*loss))
    
    opt_init, opt_update, get_params = optimizers.adam(opt.learning_rate)

    opt_update = jit(opt_update)
            
    _, params = init_fn(key, (-1, train_data[0].shape[-1]))
    opt_state = opt_init(params)

    
    train_loss = []
    train_psnr = []
    val_loss = []
    val_psnr = []

    val_loss_smooth = []
    
    xs = []

    eig_Hessian = []


    nstore = 600
    test_image_store = []
    test_image_store_PSNR = []

    for i in tqdm(range(opt.epoch)):
        

        opt_state = opt_update(i, model_grad_loss(get_params(opt_state), train_data[0], train_data[1]), opt_state)

        if i % opt.loss_record == 0:
            temp_train_loss = model_loss(get_params(opt_state), *train_data)
            temp_val_loss = model_loss(get_params(opt_state), *val_data)
            train_loss.append(temp_train_loss)
            train_psnr.append(model_psnr(temp_train_loss))
            val_loss.append(temp_val_loss)
            val_psnr.append(model_psnr(temp_val_loss))
            xs.append(i)


        if (i) % (opt.loss_record * opt.loss_smooth) == 0:
            val_loss_smooth.append(np.mean(np.array(val_loss[-opt.loss_smooth:])))

            if len(val_loss_smooth)>1:
                rel_error = np.abs((val_loss_smooth[-2]-val_loss_smooth[-1])/val_loss_smooth[-2])
                if rel_error < opt.rel_error:
                    break
                    
        if opt.epoch-i < nstore and i%50==0:
            test_pred = model_pred(get_params(opt_state), test_data[0])
            test_image_store.append(test_pred)

            test_loss =  model_loss(get_params(opt_state), *test_data)  
            test_image_store_PSNR.append(model_psnr(test_loss))
            
            

                    
    e_time = time.time()
    
    test_loss =  model_loss(get_params(opt_state), *test_data)  
    test_psnr = model_psnr(test_loss)
    
    train_pred = model_pred(get_params(opt_state), train_data[0])
    val_pred = model_pred(get_params(opt_state), val_data[0])
    test_pred = model_pred(get_params(opt_state), test_data[0])

    print(f"shape test: {np.shape(test_pred)}")
        

    history =  {
        'params': get_params(opt_state), 
        'xs': xs,
        'train_pred': train_pred,
        'val_pred': val_pred,
        'test_pred': test_pred,
        'train_loss': train_loss,
        'train_psnr': train_psnr,
        'val_loss': val_loss,
        'val_psnr': val_psnr,
        'test_loss': test_loss,
        'test_psnr': test_psnr,
        'test_image_store': test_image_store,
        'test_image_store_PSNR': test_image_store_PSNR, 
        'nstore': nstore,
        'train_time': e_time - s_time
    }

    picklename = 'results/SGDLacti%s_epoch%d_learningrate%.4e_trainloss%.4e_valloss%.4e_testloss%.4e.pickle' %(
        opt.activation, opt.epoch, opt.learning_rate, history['train_loss'][-1], history['val_loss'][-1], history['test_loss']
        )
    
    with open(picklename, 'wb') as f:
        pickle.dump([history, opt], f)  


    return 


def analysis(filepath):


    with open(filepath, 'rb') as f:
        [history, opt] = pickle.load(f)


    data, _ = data_setup(opt)

    # start = 0
    # end = 100

    nshow = 10

    # plt.figure(figsize=(10, 6))
    plt.plot(history['xs'], history['train_loss'], color="tab:blue", label='Training loss')
    plt.plot(history['xs'], history['val_loss'], color="tab:orange", linestyle="--", label='Validation loss')
    plt.xlabel('Epochs', fontsize=20)
    plt.ylabel('Loss', fontsize=20)
    plt.title(f"Single-Grade", fontsize=20)
    plt.yscale('log')
    plt.ylim([1.5e-3, 5e-2])
    plt.legend(fontsize=20)             
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    fig_filename = f'Fig/SingleGrade_deepcameraman_Loss.png'
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.savefig(fig_filename, format='png', bbox_inches='tight', pad_inches=0.1)
    plt.show()


    # plt.figure(figsize=(10, 6))
    plt.plot(history['xs'], history['train_psnr'], color="tab:blue", label='Training psnr')
    plt.plot(history['xs'], history['val_psnr'], color="tab:orange", linestyle="--", label='Validation psnr')
    plt.xlabel('Epochs', fontsize=20)
    plt.ylabel('PSNR', fontsize=20)
    plt.title(f"Single-Grade", fontsize=20)
    # plt.yscale('log')
    # plt.ylim([15, 25])
    plt.legend(fontsize=20)             
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    fig_filename = f'Fig/SingleGrade_deepcameraman_PSNR.png'
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.savefig(fig_filename, format='png', bbox_inches='tight', pad_inches=0.1)
    plt.show()

    print(f"train time: {history['train_time']}, train loss: {history['train_loss'][-1]} val loss: {history['val_loss'][-1]}, test loss: {history['test_loss']}")
    print(f"train time: {history['train_time']}, train loss: {history['train_psnr'][-1]} val loss: {history['val_psnr'][-1]}, test loss: {history['test_psnr']}")



    print(f"shape test_pred: {np.shape(history['test_pred'])}")

    plt.imshow(np.uint8(255*history["test_pred"]), cmap='gray')
    plt.title('Single-Grade', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    fig_filename = f'Fig/SingleGrade_deepcameraman_pred.png'
    plt.savefig(fig_filename, format='png', bbox_inches='tight', pad_inches=0.1)
    plt.show()

    plt.imshow(data['test_Y'], cmap='gray')
    fig_filename = f'Fig/SingleGrade_deepcameraman_org.png'
    plt.savefig(fig_filename, format='png', bbox_inches='tight', pad_inches=0)
    plt.show()

    for i in range(0, len(history["test_image_store"])):
        print(f'i: {i}, ')
        iter = opt.epoch - history['nstore'] + (i+1)*50 
        plt.imshow(np.uint8(255*history["test_image_store"][i]), cmap='gray')
        plt.title(f"Iter {iter}, PSNR: {history['test_image_store_PSNR'][i]:.2f}", fontsize=20)
        plt.axis('off')
        fig_filename = f'Fig/SingleGrade_deepcameraman_pred_iter{iter}.png'
        plt.savefig(fig_filename, format='png', bbox_inches='tight', pad_inches=0.1)
        plt.show()       

        
        
        

