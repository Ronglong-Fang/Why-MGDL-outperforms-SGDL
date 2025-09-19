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
from Hessian import compute_Hessian_one
from PIL import Image


def data_setup(opt):

    if opt.image == 'barbara':
        image_path = 'image/barbara.gif'  
        img = imageio.imread(image_path)/ 255.
        [nx, ny] = img.shape
        img = img[:int(nx/2), int(ny/2):]
        plt.imshow(img, cmap='gray')
        plt.show()       
        img = img[::2, ::2]
    elif opt.image == "resolutionchart":
        image_path = 'image/resolutionchart.tiff'  
        img = np.invert(imageio.imread(image_path))/255.
        img = img[::2, ::2]
    elif opt.image == "cameraman":
        image_path = 'image/cameraman.tif'  
        img = imageio.imread(image_path)/255.
        img = img[::2, ::2]
    elif opt.image == 'butterfly':
        print('I am butterfly')
        image_path = 'image/butterfly.gif'  
        img = imageio.imread(image_path)/255.
        img = img[::2, ::2]
        plt.imshow(img, cmap='gray')
        plt.show()
 

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

    print(f'the shape of train Y is: {np.shape(train_Y)}')
    print(f'the shape of test Y is: {np.shape(test_Y)}')


    data = {} 

    data['train_X_org'] = train_X
    data['train_Y_org'] = train_Y
    data['val_X_org'] = val_X
    data['val_Y_org'] = val_Y
    data['test_X_org'] = test_X
    data['test_Y_org'] = test_Y


    train_X = train_X.reshape(-1, 2)
    train_X = train_X.T
    train_Y = train_Y.reshape(-1, 1)
    train_Y = train_Y.T
    val_X = val_X.reshape(-1, 2)
    val_X = val_X.T
    val_Y = val_Y.reshape(-1, 1)
    val_Y = val_Y.T
    test_X = test_X.reshape(-1, 2)
    test_X = test_X.T
    test_Y = test_Y.reshape(-1, 1)
    test_Y = test_Y.T

    opt.ntrain = train_X.shape[1]


    
    data['train_X'] = train_X
    data['train_Y'] = train_Y
    data['val_X'] = val_X
    data['val_Y'] = val_Y
    data['test_X'] = test_X
    data['test_Y'] = test_Y



    return data, opt

def create_network(opt, grade):
    
    def model_fn(params, inputs, **kwargs):
        w1, b1 = params[0]
        z1 = np.dot(w1.T, inputs) + b1
        a1 = np.maximum(z1, 0)  
        
        w2, b2 = params[1]
        output = np.dot(w2.T, a1) + b2
        
        return output, z1, a1

    def he_init(key, shape):
        fan_in = shape[0]  # Number of input neurons
        std = np.sqrt(2.0 / fan_in)
        
        return jax.random.normal(key, shape) * std
    
    def init_params():
        key = jax.random.PRNGKey(42)
        
        key, subkey = jax.random.split(key)
        w1 = he_init(subkey, (opt.num_channel["grade"+str(grade)][0], opt.num_channel["grade"+str(grade)][1]))
        key, subkey = jax.random.split(key)
        b1 = np.zeros((opt.num_channel["grade"+str(grade)][1], 1))
    
        key, subkey = jax.random.split(key)
        w2 = he_init(subkey, (opt.num_channel["grade"+str(grade)][1], opt.num_channel["grade"+str(grade)][2]))
        key, subkey = jax.random.split(key)
        b2 = np.zeros((opt.num_channel["grade"+str(grade)][2], 1))

        params = [(w1, b1), (w2, b2)]
    
        return params


    return model_fn, init_params



# Train model with given hyperparameters and data
def train_model(opt, train_data, val_data, test_data, grade):

    key = random.PRNGKey(0)
    
    s_time = time.time() 
    model_fn, init_params = create_network(opt, grade)
    model_pred = jit(lambda params, x : model_fn(params, x)[0])   
    model_loss = jit(lambda params, x, y: .5 * np.mean((model_pred(params, x) - y) ** 2))
    model_grad_loss = jit(lambda params, x, y: grad(model_loss)(params, x, y))
    
    opt_init, opt_update, get_params = optimizers.sgd(opt.learning_rate)

    opt_update = jit(opt_update)


    params = init_params()

    opt_state = opt_init(params)
    train_loss = []
    val_loss = []
    val_loss_smooth = []

    xs = []
    eig_Hessian = []

    for i in tqdm(range(opt.epoch)):

        if i%opt.interval==0:
            params = get_params(opt_state)
            output, z1, a1 = model_fn(params, train_data[0])   
            Hessian = compute_Hessian_one(opt, grade, params, z1, a1, output, train_data[0], train_data[1])
            eigenvalues, _ = np.linalg.eigh(Hessian) 
            temp_eig = 1 - opt.learning_rate * eigenvalues
            eig_Hessian.append(temp_eig)
            
            min_eig = temp_eig[-1]

            print(f"shape Hessian: {np.shape(Hessian)}")
            print(f"epoch is {i}, eigenvalue of hessian is {eigenvalues}")

            if i>0 and min_eig < -0.95:
                break
                
                    

        opt_state = opt_update(i, model_grad_loss(get_params(opt_state), train_data[0], train_data[1]), opt_state)

        if i % opt.loss_record == 0:
            train_loss.append( model_loss(get_params(opt_state), *train_data) )
            val_loss.append( model_loss(get_params(opt_state), *val_data) )
            xs.append(i)

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

    _, _, train_features = model_fn(get_params(opt_state), train_data[0])
    _, _, val_features = model_fn(get_params(opt_state), val_data[0])
    _, _, test_features = model_fn(get_params(opt_state), test_data[0])


    
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
        'time': e_time - s_time,
        'eig_Hessian': eig_Hessian
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
        
        input_shape_x = np.shape(train_features)[1:]
        
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
            'eig_Hessian': history['eig_Hessian'],
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


    # plt.figure(figsize=(10, 6))
    for grade in range(1, opt.grade+1):
        
        history = SaveHistory['grade'+str(grade)]
        xs_record = [x + ite for x in history['xs']]
        ite = ite + history['xs'][-1]
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

        error1 = np.mean(np.array(history['val_loss'][-opt.loss_smooth:]))
        error2 = np.mean(np.array(history['val_loss'][-2*opt.loss_smooth:-opt.loss_smooth]))
        rel_error = np.abs(error1 - error2)/error2

        print(f"rel_error is {rel_error}")
            
        print(f"at grade {grade}, train time: {time}, train loss: {history['train_loss'][-1]} val loss: {history['val_loss'][-1]}, test loss: {history['test_loss']}")
        print(f"at grade {grade}, train time: {time}, train psnr: {model_psnr(history['train_loss'][-1])} val psnr: {model_psnr(history['val_loss'][-1])}, test psnr: {model_psnr(history['test_loss'])}")


    ite = 0 
    # plt.figure(figsize=(10, 6))
    for grade in range(1, opt.grade+1):
        
        history = SaveHistory['grade'+str(grade)]
        xs_record = [x + ite for x in history['xs']]
        ite = ite + history['xs'][-1]
        time = time + history['time'] 

        if grade==1:
            plt.plot(xs_record[1:], history['train_loss'][1:], color="tab:blue", label="Training loss")
            plt.plot(xs_record[1:], history['val_loss'][1:], color="tab:orange", linestyle="--", label="Validation loss")
        else:
            plt.plot(xs_record[1:], history['train_loss'][1:], color="tab:blue")
            plt.plot(xs_record[1:], history['val_loss'][1:], linestyle="--", color="tab:orange")

        # plt.title(f"Multi-Grade: Grade {grade}", fontsize=20)

        # plt.show()
            
        # print(f"at grade {grade}, train time: {time}, train loss: {history['train_loss'][-1]} val loss: {history['val_loss'][-1]}, test loss: {history['test_loss']}")
            
    plt.xlabel('Epochs', fontsize=20)
    plt.ylabel('Loss', fontsize=20)
    plt.title(f"Multi-Grade", fontsize=20)
    plt.yscale('log')
    plt.ylim([2.3e-2, 5e-2])
    plt.legend(fontsize=20)             
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    fig_filename = f'Fig/MultiGrade_resolutionEigStop_Loss.png'
    plt.savefig(fig_filename, format='png', bbox_inches='tight')
    plt.show()    


    for grade in range(1, opt.grade+1):

        history = SaveHistory['grade'+str(grade)]
        plt.imshow(np.uint8(255*history["test_acc"]).reshape(np.shape(data['test_Y_org'])), cmap='gray')
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.title(f"Multi-Grade: Grade {grade}", fontsize=20)
        fig_filename = f'Fig/MultiGrade_resolutionEigStop_predGrade{grade}.png'
        plt.savefig(fig_filename, format='png')
        plt.show()       

        # plt.imshow(history["test_acc"].reshape(np.shape(data['test_Y_org'])), cmap='gray')
        # plt.xticks(fontsize=20)
        # plt.yticks(fontsize=20)
        # plt.title(f"Multi-Grade: Grade {grade}", fontsize=20)
        # plt.show()    
        




    sIter=0
    # plt.figure(figsize=(10, 6))
    start=1
    for grade in range(1, opt.grade+1):
        
        history = SaveHistory['grade'+str(grade)]

        lenIter = history['xs'][-1] 

        Eig_dict_MAX = {}
        for j in range(nshow):
            Eig_dict_MAX['Index'+str(j)] = []
    
        for j in range(nshow):
            for i in range(start, len(history["eig_Hessian"])):
                Eig_dict_MAX['Index'+str(j)].append(history["eig_Hessian"][i][j])

        Eig_dict_MIN = {}
        for j in range(nshow):
            Eig_dict_MIN['Index'+str(j)] = []
    
        for j in range(nshow):
            for i in range(start, len(history["eig_Hessian"])):
                Eig_dict_MIN['Index'+str(j)].append(history["eig_Hessian"][i][len(history["eig_Hessian"][i])-nshow+j])
        
        for j in range(nshow):

            if grade==1:
                if len(range(sIter, sIter+lenIter, opt.interval)[start:]) == len(Eig_dict_MIN['Index'+str(nshow-j-1)]):
                    plt.plot(range(sIter, sIter+lenIter, opt.interval)[start:], np.array(Eig_dict_MIN['Index'+str(nshow-j-1)]).T, label=f"Index {j}")
                else:
                    plt.plot(range(sIter, sIter+lenIter+opt.interval, opt.interval)[start:], np.array(Eig_dict_MIN['Index'+str(nshow-j-1)]).T, label=f"Index {j}")
    

            
            else:

                if len(range(sIter, sIter+lenIter, opt.interval)[start:]) == len(Eig_dict_MIN['Index'+str(nshow-j-1)]):
                    plt.plot(range(sIter, sIter+lenIter, opt.interval)[start:], np.array(Eig_dict_MIN['Index'+str(nshow-j-1)]).T)
                elif len(range(sIter, sIter+lenIter, opt.interval)[start:]) < len(Eig_dict_MIN['Index'+str(nshow-j-1)]):
                    plt.plot(range(sIter, sIter+lenIter+opt.interval, opt.interval)[start:], np.array(Eig_dict_MIN['Index'+str(nshow-j-1)]).T)
                elif len(range(sIter, sIter+lenIter, opt.interval)[start:]) > len(Eig_dict_MIN['Index'+str(nshow-j-1)]):
                    plt.plot(range(sIter, sIter+lenIter-opt.interval, opt.interval)[start:], np.array(Eig_dict_MIN['Index'+str(nshow-j-1)]).T)

        for j in range(nshow):

            if grade==1:
                if len(range(sIter, sIter+lenIter, opt.interval)[start:]) == len(Eig_dict_MAX['Index'+str(nshow-j-1)]):
                    plt.plot(range(sIter, sIter+lenIter, opt.interval)[start:], np.array(Eig_dict_MAX['Index'+str(nshow-j-1)]).T, linestyle="--", label=f"Index {j+nshow}")
                else:
                    plt.plot(range(sIter, sIter+lenIter+opt.interval, opt.interval)[start:], np.array(Eig_dict_MAX['Index'+str(nshow-j-1)]).T, linestyle="--", label=f"Index {j+nshow}")
                    
            else:
                if len(range(sIter, sIter+lenIter, opt.interval)[start:]) == len(Eig_dict_MAX['Index'+str(nshow-j-1)]):
                    plt.plot(range(sIter, sIter+lenIter, opt.interval)[start:], np.array(Eig_dict_MAX['Index'+str(nshow-j-1)]).T, linestyle="--")
                elif len(range(sIter, sIter+lenIter, opt.interval)[start:]) < len(Eig_dict_MAX['Index'+str(nshow-j-1)]):
                    plt.plot(range(sIter, sIter+lenIter+opt.interval, opt.interval)[start:], np.array(Eig_dict_MAX['Index'+str(nshow-j-1)]).T, linestyle="--")
                elif len(range(sIter, sIter+lenIter, opt.interval)[start:]) > len(Eig_dict_MAX['Index'+str(nshow-j-1)]):
                    plt.plot(range(sIter, sIter+lenIter-opt.interval, opt.interval)[start:], np.array(Eig_dict_MAX['Index'+str(nshow-j-1)]).T, linestyle="--")

        sIter += lenIter


    plt.plot(range(0, sIter+opt.interval, opt.interval), 1 * np.ones_like(np.array(range(0, sIter+opt.interval, opt.interval))), 'r--')
    plt.plot(range(0, sIter+opt.interval, opt.interval), -1 * np.ones_like(np.array(range(0, sIter+opt.interval, opt.interval))), 'r--')
    plt.xlabel('Epochs', fontsize=20)
    plt.ylabel('Eigenvalues', fontsize=20)
    plt.ylim([-1.7, 1.2])

    plt.legend(fontsize=9, loc='upper left', bbox_to_anchor=(1, 1))
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.title('Multi-Grade: Eigenvalues of $\mathbf{I} - \eta\mathbf{H}_{\mathcal{L}}(\mathbf{W}^k)$', fontsize=17)
    fig_filename = f'Fig/MultiGrade_resolutionEigStop_Eig.png'
    plt.savefig(fig_filename, format='png', bbox_inches='tight')
    plt.show()




