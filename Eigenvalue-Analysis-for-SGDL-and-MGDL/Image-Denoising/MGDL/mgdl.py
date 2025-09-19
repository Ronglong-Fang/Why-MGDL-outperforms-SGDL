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
    key = random.PRNGKey(0)

    if opt.image == 'barbara':
        image_path = 'image/barbara.gif'  
        img = imageio.imread(image_path)/ 255.
        [nx, ny] = img.shape
        img = img[:int(nx/2), int(ny/2):]
        plt.imshow(img, cmap='gray')
        plt.show()       
        img = img[::2, ::2]
    elif opt.image == 'butterfly':
        print('I am butterfly')
        image_path = 'image/butterfly.gif'  
        img = imageio.imread(image_path)/255.
        img = img[::2, ::2]
        plt.imshow(img, cmap='gray')
        plt.show()


    img = img[::4, ::4]
        

    print(f'the shape of image is: {np.shape(img)}')
    
    plt.imshow(img, cmap='gray')
    plt.show()

    nx = img.shape[0]
    ny = img.shape[1]
    
    # Create input pixel coordinates in the unit square
    coords_x = np.linspace(0, 1, img.shape[0], endpoint=False)
    coords_y = np.linspace(0, 1, img.shape[1], endpoint=False)

    train_x = np.stack(np.meshgrid(coords_y, coords_x), -1)

    noise = opt.noise_level * random.normal(key, img.shape)
    train_y = img + noise


    print(f'the shape of train Y is: {np.shape(train_y)}')



    data = {} 

    data['train_x_org'] = train_x
    data['train_y_org'] = train_y
    data['img_org'] = img

    train_x = train_x.reshape(-1, 2)
    train_x = train_x.T
    train_y = train_y.reshape(-1, 1)
    train_y = train_y.T
    img = img.reshape(-1, 1)
    img = img.T


    opt.ntrain = train_x.shape[1]
    opt.nx = nx
    opt.ny = ny

    data['train_x'] = train_x
    data['train_y'] = train_y
    data['img'] = img


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

# @jit(static_argnames=['nx', 'ny'])
def tv_matrix_ver(nx, ny, pred_imgs):
    """Optimized matrix form of TV: compute the total variation."""
    # Compute the gradients (forward differences) for TV regularization
    pred_imgs = pred_imgs.reshape(nx, ny)
    TVver = np.diff(pred_imgs, axis=0, prepend=pred_imgs[[0], :])  # Vertical differences
    TVver = TVver.at[0,:].set(0)
    return (TVver.reshape(-1, 1)).T

# @jit(static_argnames=['nx', 'ny'])
def tv_matrix_hor(nx, ny, pred_imgs):
    """Optimized matrix form of TV: compute the total variation."""
    # Compute the gradients (forward differences) for TV regularization
    pred_imgs = pred_imgs.reshape(nx, ny)
    TVhor = np.diff(pred_imgs, axis=1, prepend=pred_imgs[:, [0]])  # Horizontal differences
    TVhor = TVhor.at[:,0].set(0)
    return (TVhor.reshape(-1, 1)).T

@jit
def prox_l1_u(u, alpha, beta, lambd, BN_Theta):
    """Compute prox_{alpha lambd / beta || ||_1}(alpha * BN_Theta + (1 - alpha) * u)"""
    temp_u = alpha * BN_Theta + (1 - alpha) * u
    return np.sign(temp_u) * np.maximum(np.abs(temp_u) - alpha * lambd / beta, 0)


# Train model with given hyperparameters and data
def train_model(opt, grade, u1, u2, acc_img, data, train_data):

    nx = opt.nx
    ny = opt.ny

    key = random.PRNGKey(0)
     
    model_fn, init_params = create_network(opt, grade)
    model_pred = jit(lambda params, x : model_fn(params, x)[0])   
    model_loss = jit(lambda params, u1, u2, x, y: 
                     .5 * np.sum((model_pred(params, x) - (y-acc_img)) ** 2) +  
                     .5 * opt.beta * np.sum((tv_matrix_ver(nx, ny, model_pred(params, x)) - (u1- tv_matrix_ver(nx, ny, acc_img)))**2) +
                     .5 * opt.beta * np.sum((tv_matrix_hor(nx, ny, model_pred(params, x)) - (u2- tv_matrix_hor(nx, ny, acc_img)))**2) +
                     opt.lambd * np.sum(np.abs(u1)) + opt.lambd * np.sum(np.abs(u2)))
    model_grad_loss = jit(lambda params, u1, u2, x, y: grad(model_loss)(params, u1, u2, x, y))
    model_psnr = jit(lambda pred_img, org_img: -10 * np.log10(np.mean((pred_img - org_img) ** 2)))
    
    opt_init, opt_update, get_params = optimizers.sgd(opt.learning_rate)

    opt_update = jit(opt_update)


    params = init_params()

    opt_state = opt_init(params)
    train_loss = []
    psnr = []


    xs = []
    eig_Hessian = []

    for i in tqdm(range(opt.epoch)):

        if opt.eig:
            if i%opt.interval==0:
                params = get_params(opt_state)
                output, z1, a1 = model_fn(params, train_data[0])   
                res_train_y = data['train_y'] - acc_img
                res_u1 = u1 - tv_matrix_ver(nx, ny, acc_img)
                res_u2 = u2 - tv_matrix_hor(nx, ny, acc_img)
                Hessian = compute_Hessian_one(opt, grade, params, z1, a1, output, res_u1, res_u2, train_data[0], res_train_y)
                eigenvalues, _ = np.linalg.eigh(Hessian) 
                temp_eig = 1 - opt.learning_rate * eigenvalues
                eig_Hessian.append(temp_eig)
            
                min_eig = temp_eig[-1]

                print(f"shape Hessian: {np.shape(Hessian)}")
                print(f"epoch is {i}, eigenvalue of hessian is {eigenvalues}")


        BN_Theta_ver = tv_matrix_ver(nx, ny, model_pred(params, train_data[0]) )
        u1 = prox_l1_u(u1, opt.alpha, opt.beta, opt.lambd, BN_Theta_ver)
        BN_Theta_hor = tv_matrix_hor(nx, ny, model_pred(params, train_data[0]) )
        u2 = prox_l1_u(u2, opt.alpha, opt.beta, opt.lambd, BN_Theta_hor)
        opt_state = opt_update(i, model_grad_loss(params, u1, u2, *train_data), opt_state)
        params = get_params(opt_state)

        if i % opt.loss_record == 0:
            train_loss.append( model_loss(get_params(opt_state), u1, u2, *train_data) )
            psnr.append(model_psnr(acc_img+model_pred(params, train_data[0]), data["img"]))
            xs.append(i)
            

    # confirm the last one is recorded
    train_loss.append(model_loss(params, u1, u2, *train_data) )
    psnr.append(model_psnr(acc_img+model_pred(params, train_data[0]), data["img"]))
    xs.append(i)

    acc_img = acc_img + model_pred(params, train_data[0])

    
    _, _, train_feature = model_fn(get_params(opt_state), train_data[0])


    if opt.eig:
        history =  {
            'params': get_params(opt_state), 
            'u1': u1,
            'u2': u2,
            'xs': xs,
            'acc_img': acc_img,
            'train_feature': train_feature,
            'train_loss': train_loss,
            'psnr': psnr,
            'eig_Hessian': eig_Hessian
            }
    else:
        history =  {
            'params': get_params(opt_state), 
            'u1': u1,
            'u2': u2,
            'xs': xs,
            'acc_img': acc_img,
            'train_feature': train_feature,
            'train_loss': train_loss,
            'psnr': psnr
            }
    

    return history



def MGDLmodel(opt, data):
    
    
    train_feature = data["train_x"]
    train_y = data["train_y"]
    acc_img = np.zeros_like(data['img'])
    u1 = np.zeros_like(data['img'])
    u2 = np.zeros_like(data['img'])

    SaveHistory = {}


    for grade in range(1, opt.grade+1):
        
        train_data = [train_feature, train_y]

        s_time = time.time() 
        history = train_model(opt, grade, u1, u2, acc_img, data, train_data)
        e_time = time.time()
        
        train_feature = history['train_feature']
        u1 = history['u1']
        u2 = history['u2']
        acc_img = history['acc_img']

        
        if opt.eig:
            SaveHistory['grade'+str(grade)] = {
                'params': history['params'],
                'train_loss': history['train_loss'],
                'psnr': history['psnr'],
                'acc_img': history['acc_img'],
                'eig_Hessian': history['eig_Hessian'],
                'xs': history['xs'],
                'time': e_time - s_time
            }
        else:
            SaveHistory['grade'+str(grade)] = {
                'params': history['params'],
                'train_loss': history['train_loss'],
                'psnr': history['psnr'],
                'acc_img': history['acc_img'],
                'xs': history['xs'],
                'time': e_time - s_time
            }            
        
        print(f"At grade {grade}, train time: {e_time - s_time},  train loss: {history['train_loss'][-1]}\n")
        

        
    picklename = 'results/MGDLacti%s_grade%d_epoch%d_learningrate%.2e_beta%.2e_lambd%.2e_trainloss%.2e_trainpsnr%.2e.pickle' %(
        opt.activation, opt.grade, opt.epoch, opt.learning_rate, opt.beta, opt.lambd, history['train_loss'][-1], history['psnr'][-1]
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
        # print(history['xs'])
        xs_record = [x + ite for x in history['xs']]
        ite = ite + history['xs'][-1]
        time = time + history['time'] 

        if grade==1:
            plt.plot(xs_record[1:], history['train_loss'][1:], color="tab:blue", label="Training loss")
        else:
            plt.plot(xs_record[1:], history['train_loss'][1:], color="tab:blue")

        plt.yscale('log')
        plt.title(f"Multi-Grade: Grade {grade}", fontsize=20)

        plt.show()
            
        print(f"at grade {grade}, train time: {time}, train loss: {history['train_loss'][-1]} psnr: {history['psnr'][-1]}")

    ite = 0 
    # plt.figure(figsize=(10, 6))
    for grade in range(1, opt.grade+1):
        
        history = SaveHistory['grade'+str(grade)]
        xs_record = [x + ite for x in history['xs']]
        ite = ite + history['xs'][-1]
        time = time + history['time'] 

        if grade==1:
            plt.plot(xs_record[1:], history['train_loss'][1:], color="tab:blue") 
        else:
            plt.plot(xs_record[1:], history['train_loss'][1:], color="tab:blue")

        # plt.title(f"Multi-Grade: Grade {grade}", fontsize=20)

        # plt.show()
            
        # print(f"at grade {grade}, train time: {time}, train loss: {history['train_loss'][-1]} val loss: {history['val_loss'][-1]}, test loss: {history['test_loss']}")
            
    plt.xlabel('Epochs', fontsize=20)
    plt.ylabel('Loss', fontsize=20)
    plt.title(f"Multi-Grade", fontsize=20)
    # plt.yscale('log')
    plt.ylim([32, 45])
    # plt.legend(fontsize=20)             
    plt.xticks([0, 1e5, 2e5, 3e5, 4e5], fontsize=20)
    plt.yticks(fontsize=20)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    fig_filename = f'Fig/MultiGrade_denoising10_butterfly_Loss_Eig.png'
    plt.savefig(fig_filename, format='png', bbox_inches='tight')
    plt.show()    




    for grade in range(1, opt.grade+1):
        
        history = SaveHistory['grade'+str(grade)]
        # print(history['xs'])
        xs_record = [x + ite for x in history['xs']]
        ite = ite + history['xs'][-1]
        time = time + history['time'] 

        if grade==1:
            plt.plot(xs_record[1:], history['psnr'][1:], color="tab:blue", label="Training loss")
        else:
            plt.plot(xs_record[1:], history['psnr'][1:], color="tab:blue")

        plt.yscale('log')
        plt.title(f"Multi-Grade: Grade {grade}", fontsize=20)
        plt.show()


    ite = 0 
    # plt.figure(figsize=(10, 6))
    for grade in range(1, opt.grade+1):
        
        history = SaveHistory['grade'+str(grade)]
        xs_record = [x + ite for x in history['xs']]
        ite = ite + history['xs'][-1]
        time = time + history['time'] 

        if grade==1:
            plt.plot(xs_record[1:], history['psnr'][1:], color="tab:blue") 
        else:
            plt.plot(xs_record[1:], history['psnr'][1:], color="tab:blue")

        # plt.title(f"Multi-Grade: Grade {grade}", fontsize=20)

        # plt.show()
            
        # print(f"at grade {grade}, train time: {time}, train loss: {history['train_loss'][-1]} val loss: {history['val_loss'][-1]}, test loss: {history['test_loss']}")
            
    plt.xlabel('Epochs', fontsize=20)
    plt.ylabel('Loss', fontsize=20)
    plt.title(f"Multi-Grade", fontsize=20)
    plt.yscale('log')
    # plt.ylim([5e-3, 1.5e-2])
    # plt.legend(fontsize=20)             
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    plt.show()    
            



    for grade in range(1, opt.grade+1):

        history = SaveHistory['grade'+str(grade)]
        plt.imshow(history["acc_img"].reshape(np.shape(data['train_y_org'])), cmap='gray')
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.title(f"Multi-Grade: Grade {grade}", fontsize=20)
        fig_filename = f'Fig/MultiGrade_denoising10_butterfly_predGrade{grade}.png'
        plt.savefig(fig_filename, format='png')
        plt.show()       
        



    if opt.eig:
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
                    else:
                        plt.plot(range(sIter, sIter+lenIter+opt.interval, opt.interval)[start:], np.array(Eig_dict_MIN['Index'+str(nshow-j-1)]).T)
    
            for j in range(nshow):
    
                if grade==1:
                    plt.plot(range(sIter, sIter+lenIter, opt.interval)[start:], np.array(Eig_dict_MAX['Index'+str(nshow-j-1)]).T, linestyle="--", label=f"Index {j+nshow}")
                else:
                    if len(range(sIter, sIter+lenIter, opt.interval)[start:]) == len(Eig_dict_MIN['Index'+str(nshow-j-1)]):
                        plt.plot(range(sIter, sIter+lenIter, opt.interval)[start:], np.array(Eig_dict_MIN['Index'+str(nshow-j-1)]).T)
                    else:
                        plt.plot(range(sIter, sIter+lenIter+opt.interval, opt.interval)[start:], np.array(Eig_dict_MIN['Index'+str(nshow-j-1)]).T)
    
            sIter += lenIter
    
    
        plt.plot(range(0, sIter, opt.interval), 1 * np.ones_like(np.array(range(0, sIter, opt.interval))), 'r--')
        plt.plot(range(0, sIter, opt.interval), -1 * np.ones_like(np.array(range(0, sIter, opt.interval))), 'r--')
        plt.xlabel('Epochs', fontsize=20)
        plt.ylabel('Eigenvalues', fontsize=20)
        plt.ylim([-1.2, 1.2])
    
        plt.legend(fontsize=9, loc='upper left', bbox_to_anchor=(1, 1))
        plt.xticks([0, 1e5, 2e5, 3e5, 4e5], fontsize=20)
        plt.yticks(fontsize=20)
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        plt.title('Multi-Grade: Eigenvalues of $\mathbf{I} - \eta\mathbf{H}_{\mathcal{L}}(\mathbf{W}^k)$', fontsize=17)
        fig_filename = f'Fig/MultiGrade_denoising10_butterfly_Eig.png'
        plt.savefig(fig_filename, format='png', bbox_inches='tight')
        plt.show()
    
    
    
    
     
