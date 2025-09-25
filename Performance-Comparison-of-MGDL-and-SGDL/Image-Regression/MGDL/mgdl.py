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
    
    plt.imshow(np.uint8(255*img), cmap='gray')
    plt.axis('off')
    fig_filename = f'Fig/MultiGrade_deepresolutionchart_org.png'
    plt.savefig(fig_filename, format='png', bbox_inches='tight', pad_inches=0.1)
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
def create_network(opt, grade):

    layers = []

    for i in range(len(opt.num_channel['grade'+str(grade)])-2):
        layers.append(stax.Dense(opt.num_channel['grade'+str(grade)][i+1]))
        layers.append(stax.Relu)

    # Final dense layer for x
    _, feature_fn = stax.serial(*layers)
    
    layers.append(stax.Dense(opt.num_channel['grade'+str(grade)][-1]))
    layers.append(stax.Identity)

    init_fn, model_fn = stax.serial(*layers)
        
    return feature_fn, model_fn, init_fn



# Train model with given hyperparameters and data
def train_model(opt, train_data, val_data, test_data, grade):

    key = random.PRNGKey(0)
    
    s_time = time.time() 
    feature_fn, model_fn, init_fn = create_network(opt, grade)

    
    model_pred = jit(lambda params, x : np.squeeze(model_fn(params, x))) 
    feature_pred = jit(lambda params, x : feature_fn(params, x))
    model_loss = jit(lambda params, x, y: .5 * np.mean((model_pred(params, x) - y) ** 2))
    model_grad_loss = jit(lambda params, x, y: grad(model_loss)(params, x, y))
    model_psnr = jit(lambda loss: -10*np.log10(2.*loss))
    
    opt_init, opt_update, get_params = optimizers.adam(opt.learning_rate)

    opt_update = jit(opt_update)


    _, params = init_fn(key, (-1, train_data[0].shape[-1]))

    opt_state = opt_init(params)
    train_loss = []
    val_loss = []
    val_loss_smooth = []
    test_image_store = []
    test_image_store_PSNR = []

    nstore = 500

    xs = []

    for i in tqdm(range(opt.epoch)):
                    

        opt_state = opt_update(i, model_grad_loss(get_params(opt_state), train_data[0], train_data[1]), opt_state)

        if i % opt.loss_record == 0:
            train_loss.append( model_loss(get_params(opt_state), *train_data) )
            val_loss.append( model_loss(get_params(opt_state), *val_data) )
            xs.append(i)

        if grade==4 and i%50==0:
            test_pred = model_pred(get_params(opt_state), test_data[0])
            test_image_store.append(test_pred)
    
            test_loss =  model_loss(get_params(opt_state), *test_data)  
            test_image_store_PSNR.append(model_psnr(test_loss))


        if (i+1) % (opt.loss_record * opt.loss_smooth) == 0:
            val_loss_smooth.append(np.mean(np.array(val_loss[-opt.loss_smooth:])))

            if len(val_loss_smooth)>1:
                rel_error = np.abs((val_loss_smooth[-2]-val_loss_smooth[-1])/val_loss_smooth[-2])
                if rel_error < opt.rel_error:
                    break


                    
    e_time = time.time()
    
    test_loss =  model_loss(get_params(opt_state), *test_data)

    train_pred = model_pred(get_params(opt_state), train_data[0])
    val_pred = model_pred(get_params(opt_state), val_data[0])
    test_pred = model_pred(get_params(opt_state), test_data[0])

    train_features = feature_pred(get_params(opt_state), train_data[0])
    val_features = feature_pred(get_params(opt_state), val_data[0])
    test_features = feature_pred(get_params(opt_state), test_data[0])

    print(f"shape train pred: {np.shape(train_pred)}")
    print(f"shape test pred: {np.shape(test_pred)}")





    
    history =  {
        'params': get_params(opt_state), 
        'xs': xs,
        'train_pred': train_pred,
        'val_pred': val_pred,
        'test_pred': test_pred,
        'train_features': train_features,
        'val_features': val_features,
        'test_features': test_features,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'test_loss': test_loss,
        'test_image_store': test_image_store,
        'test_image_store_PSNR': test_image_store_PSNR, 
        'time': e_time - s_time
    }

    return history



def MGDLmodel(opt, data):
    
    
    train_features = data["train_X"]
    val_features = data["val_X"]
    test_features = data["test_X"]

    train_Y = data["train_Y"]
    val_Y = data["val_Y"]
    test_Y = data["test_Y"]
    
    train_acc = np.zeros_like(data["train_Y"])
    val_acc = np.zeros_like(data["val_Y"])
    test_acc = np.zeros_like(data["test_Y"])

    SaveHistory = {}


    for grade in range(1, opt.grade+1):
        
        train_data = [train_features, train_Y]
        val_data = [val_features, val_Y]
        test_data = [test_features, test_Y]
        
        s_time = time.time() 
        history = train_model(opt, train_data, val_data, test_data, grade)
        e_time = time.time()
        
        train_features = history['train_features']
        val_features = history['val_features']
        test_features = history['test_features']

        train_Y -= history['train_pred']
        val_Y -= history['val_pred']
        test_Y -= history['test_pred']

        train_acc += history['train_pred']
        val_acc += history['val_pred']
        test_acc += history['test_pred']

        
        SaveHistory['grade'+str(grade)] = {
            'params': history['params'],
            'train_loss': history['train_loss'],
            'val_loss': history['val_loss'],
            'test_loss': history['test_loss'],
            'test_image_store': history['test_image_store'],
            'test_image_store_PSNR': history['test_image_store_PSNR'], 
            'train_acc': train_acc,
            'val_acc': val_acc,
            'test_acc': test_acc,
            'xs': history['xs'],
            'time': e_time - s_time
        }
        
        print(f"At grade {grade}, train time: {e_time - s_time},  train loss: {history['train_loss'][-1]}, val loss: {history['val_loss'][-1]}, test loss: {history['test_loss']}\n")
        

        
    picklename = 'results/MGDLacti%s_grade%d_epoch%d_learningrate%.2e_trainloss%.2e_valloss%.2e_testloss%.2e.pickle' %(
        opt.activation, opt.grade, opt.epoch, opt.learning_rate, history['train_loss'][-1], history['val_loss'][-1], history['test_loss']
        )
    
    with open(picklename, 'wb') as f:
        pickle.dump([SaveHistory, opt], f)      





def analysis(filepath):

    model_psnr = jit(lambda loss: -10*np.log10(2.*loss))


    with open(filepath, 'rb') as f:
        [SaveHistory, opt] = pickle.load(f)


    data, _ = data_setup(opt)   


    nshow = 10

    ite = 0 
    time = 0


    # opt.grade = 3


    # plt.figure(figsize=(10, 6))
    for grade in range(1, opt.grade+1):
        
        history = SaveHistory['grade'+str(grade)]
        xs_record = [x + ite for x in history['xs']]
        ite = ite + opt.epoch
        time = time + history['time'] 

        if grade==1:
            plt.plot(xs_record[1:], history['train_loss'][1:], color="tab:blue", label="Training loss")
            plt.plot(xs_record[1:], history['val_loss'][1:], color="tab:orange", linestyle="--", label="Validation loss")
        else:
            plt.plot(xs_record[1:], history['train_loss'][1:], color="tab:blue")
            plt.plot(xs_record[1:], history['val_loss'][1:], linestyle="--", color="tab:orange")

        plt.yscale('log')
        plt.title(f"Multi-Grade: Grade {grade}", fontsize=20)

        plt.show()
            
        print(f"at grade {grade}, train time: {time}, train loss: {history['train_loss'][-1]} val loss: {history['val_loss'][-1]}, test loss: {history['test_loss']}")

        print(f"at grade {grade}, train time: {time}, train psnr: {model_psnr(history['train_loss'][-1])} val psnr: {model_psnr(history['val_loss'][-1])}, test psnr: {model_psnr(history['test_loss'])}")


    ite = 0 
    MUL_EPOCH = [0]
    # plt.figure(figsize=(10, 6))
    for grade in range(1, opt.grade+1):
        
        history = SaveHistory['grade'+str(grade)]
        xs_record = [x + ite for x in history['xs']]
        ite = ite + history['xs'][-1]
        MUL_EPOCH.append(ite)
        time = time + history['time'] 

        if grade==1:
            plt.plot(xs_record, history['train_loss'], color="tab:blue", label="Training loss")
            plt.plot(xs_record, history['val_loss'], color="tab:orange", linestyle="--", label="Validation loss")
        else:
            plt.plot(xs_record, history['train_loss'], color="tab:blue")
            plt.plot(xs_record, history['val_loss'], linestyle="--", color="tab:orange")

            
    plt.xlabel('Epochs', fontsize=20)
    plt.ylabel('Loss', fontsize=20)
    plt.title(f"Multi-Grade", fontsize=20)
    plt.yscale('log')
    plt.ylim([1.5e-3, 5e-2])
    plt.legend(fontsize=20)             
    for x in MUL_EPOCH:
        plt.axvline(x, color='k', linestyle=':')
    plt.xticks([0, 1e4, 2e4, 3e4]) 
    plt.xticks(fontsize=20) 
    plt.yticks(fontsize=20)
    fig_filename = f'Fig/MultiGrade_deepcameraman_Loss.png'
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.savefig(fig_filename, format='png', bbox_inches='tight', pad_inches=0.1)
    plt.show()    


    ite = 0 
    # plt.figure(figsize=(10, 6))
    for grade in range(1, opt.grade+1):
        
        history = SaveHistory['grade'+str(grade)]
        xs_record = [x + ite for x in history['xs']]
        ite = ite + history['xs'][-1]
        time = time + history['time'] 

        if grade==1:
            plt.plot(xs_record, model_psnr(np.array(history['train_loss'])), color="tab:blue", label="Training loss")
            plt.plot(xs_record, model_psnr(np.array(history['val_loss'])), color="tab:orange", linestyle="--", label="Validation loss")
        else:
            plt.plot(xs_record, model_psnr(np.array(history['train_loss'])), color="tab:blue")
            plt.plot(xs_record, model_psnr(np.array(history['val_loss'])), linestyle="--", color="tab:orange")

            
    plt.xlabel('Epochs', fontsize=20)
    plt.ylabel('PSNR', fontsize=20)
    plt.title(f"Multi-Grade", fontsize=20)
    # plt.yscale('log')
    # plt.ylim([15, 25])
    plt.legend(fontsize=20)             
    for x in MUL_EPOCH:
        plt.axvline(x, color='k', linestyle=':')
    plt.xticks([0, 1e4, 2e4, 3e4])
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    fig_filename = f'Fig/MultiGrade_deepcameraman_PSNR.png'
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.savefig(fig_filename, format='png', bbox_inches='tight', pad_inches=0.1)
    plt.show()    


    for grade in range(1, opt.grade+1):

        history = SaveHistory['grade'+str(grade)]
        print(f"shape train acc: {np.shape(history['train_acc'])}")
        print(f"shape test acc: {np.shape(history['test_acc'])}")
        # print(f"shape train features: {np.shape(train_features)}")
        plt.imshow(np.uint8(255*history["test_acc"]), cmap='gray')
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.title(f"Multi-Grade: Grade {grade}", fontsize=20)
        fig_filename = f'Fig/MultiGrade_deepcameraman_predGrade{grade}.png'
        plt.savefig(fig_filename, format='png', bbox_inches='tight', pad_inches=0.1)
        plt.show()       


    
    history = SaveHistory['grade'+str(4)]
    print(len(history["test_image_store"]))
    for i in range(len(history["test_image_store"])-5, len(history["test_image_store"])):
        print(f'i: {i}, ')
        iter = i*50 
        plt.imshow(np.uint8(255*(history["test_image_store"][i] + SaveHistory['grade'+str(3)]['test_acc'])), cmap='gray')
        plt.title(f"Iter {iter}, PSNR: {history['test_image_store_PSNR'][i]:.2f}", fontsize=20)
        plt.axis('off')
        fig_filename = f'Fig/MultiGrade_deepcameraman_predGrade{grade}_iter{iter}.png'
        plt.savefig(fig_filename, format='png', bbox_inches='tight', pad_inches=0.1)
        plt.show()       





 