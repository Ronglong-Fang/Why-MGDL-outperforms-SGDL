import jax.numpy as np


def relu_grad(x):
    return np.where(x >= 0, 1, 0)

def compute_Hessian_two(opt, params, z1, z2, a1, a2, output, train_x, train_y):

    W1, b1 = params[0]                      
    W2, b2 = params[1]  
    W3, b3 = params[2]

    
    input_dim = opt.num_channel[0]
    m1 = opt.num_channel[1]
    m2 = opt.num_channel[2]
    
    grad_W1W1 = np.zeros((input_dim * m1, input_dim * m1))
    grad_W1b1 = np.zeros((input_dim * m1, m1))
    grad_W1W2 = np.zeros((input_dim * m1, m1*m2))
    grad_W1b2 = np.zeros((input_dim * m1, m2))
    grad_W1W3 = np.zeros((input_dim * m1, m2))
    grad_W1b3 = np.zeros((input_dim * m1, 1))
    
    grad_b1b1 = np.zeros((m1, m1))
    grad_b1W2 = np.zeros((m1, m1*m2))
    grad_b1b2 = np.zeros((m1, m2))
    grad_b1W3 = np.zeros((m1, m2))
    grad_b1b3 = np.zeros((m1, 1))

    grad_W2W2 = np.zeros((m1*m2, m1*m2))
    grad_W2b2 = np.zeros((m1*m2, m2))
    grad_W2W3 = np.zeros((m1*m2, m2))
    grad_W2b3 = np.zeros((m1*m2, 1))

    grad_b2b2 = np.zeros((m2, m2))
    grad_b2W3 = np.zeros((m2, m2))
    grad_b2b3 = np.zeros((m2, 1))

    grad_W3W3 = np.zeros((m2, m2))
    grad_W3b3 = np.zeros((m2, 1))

    q1 = 1/opt.ntrain
    
    for i in range(opt.ntrain):
        x_i = train_x[:,i]
        y_i = train_y[:,i]
        z1_i = z1[:, i]
        z1_i = z1_i.reshape(np.shape(z1_i)[0], 1)
        z2_i = z2[:, i]
        z2_i = z2_i.reshape(np.shape(z2_i)[0], 1)
        a1_i = a1[:, i]
        a1_i = a1_i.reshape(np.shape(a1_i)[0], 1)
        a2_i = a2[:, i]
        a2_i = a2_i.reshape(np.shape(a2_i)[0], 1)
        output_i = output[:, i]
        output_i = output_i.reshape(np.shape(output_i)[0], 1)

        e_i = output_i - y_i

        #------------grad_W1W1-------------------
        grad_diag_z1 = np.diag(relu_grad(z1_i.flatten()))
        grad_diag_z2 = np.diag(relu_grad(z2_i.flatten()))
        temp1 = (np.dot(grad_diag_z1, W2)).reshape((m1, m2))                             #(m1, m2)
        temp2 = (np.dot(grad_diag_z2, W3)).reshape((m2, 1))                              #(m2, 1)
        temp3 = np.dot(temp1, temp2)                                                                             #(m1, 1)

        #-----------
        grad_W1W1 += q1 * np.kron(np.dot(temp3, temp3.T), np.dot(x_i, x_i.T))                                           #(d*m1, d*m1)
        grad_W1b1 += q1 * np.kron(np.dot(temp3, temp3.T), x_i)                                                          #(d*m1, m1)
        grad_W1W2 += (q1 * np.dot(np.kron(temp3, x_i), np.kron(temp2.T, a1_i.T)) + 
                      q1 * np.kron(np.kron(temp2.T, grad_diag_z1),  x_i) * e_i)                #(d*m1, m1*m2)
        grad_W1b2 += q1 * np.dot(np.kron(temp3,  x_i), temp2.T)                                                         #(d*m1, m2)
        grad_W1W3 += (q1 * np.dot(np.kron(temp3, x_i), a2_i.T) + 
                      q1 * np.kron(np.dot(temp1, grad_diag_z2),  x_i) * e_i)                   #(d*m1, m2)
        grad_W1b3 += q1 * np.kron(temp3, x_i)                                                                           #(d*m1, 1)
        
        #-----------                                                                                            
        grad_b1b1 += q1 * np.dot(temp3, temp3.T)                                                                        #(m1, m1)
        grad_b1W2 += (q1 * np.dot(temp3, np.kron(temp2.T, a1_i.T)) +
                      q1 * np.kron(temp2.T, grad_diag_z1) * e_i)                                #(m1, m1*m2) 
        grad_b1b2 += q1 * np.dot(temp3, temp2.T)                                                                         #(m1, m2)
        grad_b1W3 += (q1 * np.dot(temp3, a2_i.T) +
                      q1 * np.dot(temp1, grad_diag_z2) * e_i)                                   #(m1, m2)
        grad_b1b3 += q1 * temp3                                                                                          #(m1, 1)

        #-----------
        grad_W2W2 += q1 * np.dot(np.kron(temp2, a1_i),  np.kron(temp2.T, a1_i.T))                                        #(m1*m2, m1*m2)  
        grad_W2b2 += q1 * np.dot(np.kron(temp2, a1_i), temp2.T)                                                          #(m1*m2, m2)   
        grad_W2W3 += (q1 * np.dot(np.kron(temp2, a1_i),  a2_i.T) +
                     q1 * np.kron(grad_diag_z2, a1_i) * e_i)                                    #(m1*m2, m2)   
        grad_W2b3 += q1 * np.kron(temp2, a1_i)                                                                           #(m1*m2, 1)  

        #-----------
        grad_b2b2 += q1 * np.dot(temp2, temp2.T)                                                                         #(m2, m2)   
        grad_b2W3 += (q1 * np.dot(temp2,  a2_i.T) +
                     q1 * grad_diag_z2 * e_i)                                                   #(m2, m2)    
        grad_b2b3 += q1 * temp2                                                                                          #(m2, 1)

        #-----------
        grad_W3W3 += q1 * np.dot(a2_i, a2_i.T)                                                                          #(m2, m2)
        grad_W3b3 += q1 * a2_i                                                                                          #(m2, 1)      
        
        
    grad_b1W1 = grad_W1b1.T                                                                                    #(m1, d*m1)
    
    grad_W2W1 = grad_W1W2.T                                                                                    #(m1*m2, d*m1)
    grad_W2b1 = grad_b1W2.T                                                                                    #(m1*m2, m1)

    grad_b2W1 = grad_W1b2.T                                                                                    #(m2, d*m1)
    grad_b2b1 = grad_b1b2.T                                                                                    #(m2, m1)
    grad_b2W2 = grad_W2b2.T                                                                                    #(m2, m1*m2)

    grad_W3W1 = grad_W1W3.T                                                                                    #(m2, d*m1)  
    grad_W3b1 = grad_b1W3.T                                                                                    #(m2, m1)
    grad_W3W2 = grad_W2W3.T                                                                                    #(m2, m1*m2)
    grad_W3b2 = grad_b2W3.T                                                                                    #(m2, m2)

    grad_b3W1 = grad_W1b3.T                                                                                    #(1, d*m1)
    grad_b3b1 = grad_b1b3.T                                                                                    #(1, m1)
    grad_b3W2 = grad_W2b3.T                                                                                    #(1, m1*m2)
    grad_b3b2 = grad_b2b3.T                                                                                    #(1, m2)
    grad_b3W3 = grad_W3b3.T                                                                                    #(1, m2)
    grad_b3b3 = 1                                                                                              #(1, 1)  
    
    
    
    Hessian = np.block([[grad_W1W1, grad_W1b1, grad_W1W2, grad_W1b2, grad_W1W3, grad_W1b3],
                        [grad_b1W1, grad_b1b1, grad_b1W2, grad_b1b2, grad_b1W3, grad_b1b3],
                        [grad_W2W1, grad_W2b1, grad_W2W2, grad_W2b2, grad_W2W3, grad_W2b3],
                        [grad_b2W1, grad_b2b1, grad_b2W2, grad_b2b2, grad_b2W3, grad_b2b3],
                        [grad_W3W1, grad_W3b1, grad_W3W2, grad_W3b2, grad_W3W3, grad_W3b3],
                        [grad_b3W1, grad_b3b1, grad_b3W2, grad_b3b2, grad_b3W3, grad_b3b3]])
    
    return Hessian



def compute_Hessian_three(opt, params, z1, z2, z3, a1, a2, a3, output, train_x, train_y):


    W1, b1 = params[0]                      
    W2, b2 = params[1]  
    W3, b3 = params[2]
    W4, b4 = params[3]

    
    input_dim = opt.num_channel[0]
    m1 = opt.num_channel[1]
    m2 = opt.num_channel[2]
    m3 = opt.num_channel[3]

    
    grad_W1W1 = np.zeros((input_dim * m1, input_dim * m1))
    grad_W1b1 = np.zeros((input_dim * m1, m1))
    grad_W1W2 = np.zeros((input_dim * m1, m1*m2))
    grad_W1b2 = np.zeros((input_dim * m1, m2))
    grad_W1W3 = np.zeros((input_dim * m1, m2*m3))
    grad_W1b3 = np.zeros((input_dim * m1, m3))
    grad_W1W4 = np.zeros((input_dim * m1, m3))
    grad_W1b4 = np.zeros((input_dim * m1, 1))
    
    grad_b1b1 = np.zeros((m1, m1))
    grad_b1W2 = np.zeros((m1, m1*m2))
    grad_b1b2 = np.zeros((m1, m2))
    grad_b1W3 = np.zeros((m1, m2*m3))
    grad_b1b3 = np.zeros((m1, m3))
    grad_b1W4 = np.zeros((m1, m3))
    grad_b1b4 = np.zeros((m1, 1))

    grad_W2W2 = np.zeros((m1*m2, m1*m2))
    grad_W2b2 = np.zeros((m1*m2, m2))
    grad_W2W3 = np.zeros((m1*m2, m2*m3))
    grad_W2b3 = np.zeros((m1*m2, m3))
    grad_W2W4 = np.zeros((m1*m2, m3))
    grad_W2b4 = np.zeros((m1*m2, 1))

    grad_b2b2 = np.zeros((m2, m2))
    grad_b2W3 = np.zeros((m2, m2*m3))
    grad_b2b3 = np.zeros((m2, m3))
    grad_b2W4 = np.zeros((m2, m3))
    grad_b2b4 = np.zeros((m2, 1))

    grad_W3W3 = np.zeros((m2*m3, m2*m3))
    grad_W3b3 = np.zeros((m2*m3, m3))
    grad_W3W4 = np.zeros((m2*m3, m3))
    grad_W3b4 = np.zeros((m2*m3, 1))

    grad_b3b3 = np.zeros((m3, m3))
    grad_b3W4 = np.zeros((m3, m3))
    grad_b3b4 = np.zeros((m3, 1))

    grad_W4W4 = np.zeros((m3, m3))
    grad_W4b4 = np.zeros((m3, 1))

    q1 = 1/opt.ntrain

    
    for i in range(opt.ntrain):
        x_i = train_x[:,i]
        y_i = train_y[:,i]
    
        z1_i = z1[:, i]
        z1_i = z1_i.reshape(np.shape(z1_i)[0], 1)
        z2_i = z2[:, i]
        z2_i = z2_i.reshape(np.shape(z2_i)[0], 1)
        z3_i = z3[:, i]
        z3_i = z3_i.reshape(np.shape(z3_i)[0], 1)
        a1_i = a1[:, i]
        a1_i = a1_i.reshape(np.shape(a1_i)[0], 1)
        a2_i = a2[:, i]
        a2_i = a2_i.reshape(np.shape(a2_i)[0], 1)
        a3_i = a3[:, i]
        a3_i = a3_i.reshape(np.shape(a3_i)[0], 1)
        output_i = output[:, i]
        output_i = output_i.reshape(np.shape(output_i)[0], 1)

        e_i = output_i - y_i

        #------------grad_W1W1-------------------
        grad_diag_z1 = np.diag(relu_grad(z1_i.flatten()))
        grad_diag_z2 = np.diag(relu_grad(z2_i.flatten()))
        grad_diag_z3 = np.diag(relu_grad(z3_i.flatten()))
        temp1 = (np.dot( grad_diag_z1, W2)).reshape((m1, m2))                                  #(m1, m2)
        temp2 = (np.dot( grad_diag_z2, W3)).reshape((m2, m3))                                  #(m2, m3)
        temp3 = (np.dot( grad_diag_z3, W4)).reshape((m3, 1))                                   #(m3, 1)
        
        temp4 = np.dot(np.dot(temp1, temp2), temp3)                                                                  #(m1, 1)
        temp5 = np.dot(temp2, temp3)                                                                                 #(m2, 1)
        temp6 = np.dot(temp1, temp2)                                                                                 #(m1, m3)

        #-----------
        grad_W1W1 += q1 * np.kron(np.dot(temp4, temp4.T), np.dot(x_i, x_i.T))                                           #(d*m1, d*m1)
        grad_W1b1 += q1 * np.kron(np.dot(temp4, temp4.T), x_i)                                                          #(d*m1, m1)
        grad_W1W2 += (q1 * np.dot(np.kron(temp4, x_i), np.kron(temp5.T, a1_i.T)) + 
                      q1 * np.kron(np.kron(temp5.T, grad_diag_z1),  x_i) * e_i)      #(d*m1, m1*m2)
        grad_W1b2 += q1 * np.dot(np.kron(temp4,  x_i), temp5.T)                                                         #(d*m1, m2)
        grad_W1W3 += (q1 * np.dot(np.kron(temp4, x_i), np.kron(temp3.T, a2_i.T)) + 
                      q1 * np.kron(np.kron(temp3.T, np.dot(temp1, grad_diag_z2)),  x_i) * e_i)        #(d*m1, m2*m3)
        grad_W1b3 += q1 * np.dot(np.kron(temp4, x_i), temp3.T)                                                                           #(d*m1, m3)
        grad_W1W4 += (q1 * np.dot(np.kron(temp4, x_i), a3_i.T)  + 
                      q1 * np.kron(np.dot(temp6,  grad_diag_z3), x_i) * e_i)                         #(d*m1, m3)
        grad_W1b4 += q1 * np.kron(temp4, x_i)                                                                                            #(d*m1, 1)
        
        
        #-----------                                                                                        
        grad_b1b1 += q1 * np.dot(temp4, temp4.T)                                                                         #(m1, m1)
        grad_b1W2 += (q1 * np.dot(temp4, np.kron(temp5.T, a1_i.T)) +
                      q1 * np.kron(temp5.T, grad_diag_z1) * e_i)                      #(m1, m1*m2) 
        grad_b1b2 += q1 * np.dot(temp4, temp5.T)                                                                         #(m1, m2)
        grad_b1W3 += (q1 * np.dot(temp4, np.kron(temp3.T, a2_i.T)) +
                      q1 * np.kron(temp3.T, np.dot(temp1, grad_diag_z2)) * e_i)         #(m1, m2*m3)
        grad_b1b3 += q1 * np.dot(temp4, temp3.T)                                                                         #(m1, m3)
        grad_b1W4 += (q1 * np.dot(temp4, a3_i.T) +
                      q1 * np.dot(temp6, grad_diag_z3) * e_i)                         #(m1, m3)
        grad_b1b4 += q1 * temp4  

        #-----------
        grad_W2W2 += q1 * np.dot(np.kron(temp5, a1_i),  np.kron(temp5.T, a1_i.T))                                        #(m1*m2, m1*m2)  
        grad_W2b2 += q1 * np.dot(np.kron(temp5, a1_i), temp5.T)                                                          #(m1*m2, m2)   
        grad_W2W3 += (q1 * np.dot(np.kron(temp5, a1_i), np.kron(temp3.T, a2_i.T)) +
                      q1 * np.kron(np.kron(temp3.T, grad_diag_z2), a1_i) * e_i)         #(m1*m2, m2*m3)   
        grad_W2b3 += q1 * np.dot(np.kron(temp5, a1_i), temp3.T)                                                          #(m1*m2, m3)  
        grad_W2W4 += (q1 * np.dot(np.kron(temp5, a1_i),  a3_i.T) +
                      q1 * np.kron(np.dot(temp2, grad_diag_z3), a1_i) * e_i)         #(m1*m2, m3)   
        grad_W2b4 += q1 * np.kron(temp5, a1_i)                                                                           #(m1*m2, 1)  

        #-----------
        grad_b2b2 += q1 * np.dot(temp5, temp5.T)                                                                         #(m2, m2)   
        grad_b2W3 += (q1 * np.dot(temp5,  np.kron(temp3.T, a2_i.T)) +
                      q1 * np.kron(temp3.T, grad_diag_z2) * e_i)                                            #(m2, m2*m3)    
        grad_b2b3 += q1 * np.dot(temp5, temp3.T)                                                                         #(m2, m3)
        grad_b2W4 += (q1 * np.dot(temp5,  a3_i.T) +
                      q1 * np.dot(temp2, grad_diag_z3) * e_i)                                              #(m2, m3)    
        grad_b2b4 += q1 * temp5                                                                                          #(m2, 1)

        #-----------
        grad_W3W3 += q1 * np.dot( np.kron(temp3, a2_i), np.kron(temp3.T, a2_i.T) )                                       #(m2*m3, m2*m3)
        grad_W3b3 += q1 * np.dot( np.kron(temp3, a2_i), temp3.T)                                                         #(m2, 1)    
        grad_W3W4 += (q1*np.dot(np.kron(temp3, a2_i), a3_i.T) +
                      q1*np.kron(grad_diag_z3, a2_i) * e_i)                                              #(m2, m3)    
        grad_W3b4 += q1 * np.kron(temp3, a2_i) 


        grad_b3b3 += q1 * np.dot( temp3, temp3.T)                                                                                            #(m2, 1)      
        grad_b3W4 += (q1 * np.dot(temp3, a3_i.T) +
                      q1 * grad_diag_z3 * e_i)                                              #(m2, m3)    
        grad_b3b4 += q1 * temp3

        grad_W4W4 += q1 * np.dot(a3_i, a3_i.T)                                                                #(m3, m3)    
        grad_W4b4 += q1 * a3_i                                                                                #(m3, 1)
        
    grad_b1W1 = grad_W1b1.T                                                                                 
    
    grad_W2W1 = grad_W1W2.T                                                                                    
    grad_W2b1 = grad_b1W2.T                                                                                 

    grad_b2W1 = grad_W1b2.T                                                                                   
    grad_b2b1 = grad_b1b2.T                                                                                    
    grad_b2W2 = grad_W2b2.T                                                                                    

    grad_W3W1 = grad_W1W3.T                                                                                    
    grad_W3b1 = grad_b1W3.T                                                                                    
    grad_W3W2 = grad_W2W3.T                                                                                    
    grad_W3b2 = grad_b2W3.T                                                                                   

    grad_b3W1 = grad_W1b3.T                                                                                   
    grad_b3b1 = grad_b1b3.T                                                                                  
    grad_b3W2 = grad_W2b3.T                                                                                    
    grad_b3b2 = grad_b2b3.T                                                                                  
    grad_b3W3 = grad_W3b3.T                                                                                    

    grad_W4W1 = grad_W1W4.T                                                                                    
    grad_W4b1 = grad_b1W4.T                                                                                    
    grad_W4W2 = grad_W2W4.T                                                                                    
    grad_W4b2 = grad_b2W4.T                                                                                    
    grad_W4W3 = grad_W3W4.T                                                                                      
    grad_W4b3 = grad_b3W4.T      

    grad_b4W1 = grad_W1b4.T                                                                                    
    grad_b4b1 = grad_b1b4.T                                                                                   
    grad_b4W2 = grad_W2b4.T                                                                                    
    grad_b4b2 = grad_b2b4.T                                                                                    
    grad_b4W3 = grad_W3b4.T                                                                                    
    grad_b4b3 = grad_b3b4.T    
    grad_b4W4 = grad_W4b4.T                                                                                 
    grad_b4b4 = 1

    
    
    Hessian = np.block([[grad_W1W1, grad_W1b1, grad_W1W2, grad_W1b2, grad_W1W3, grad_W1b3, grad_W1W4, grad_W1b4],
                        [grad_b1W1, grad_b1b1, grad_b1W2, grad_b1b2, grad_b1W3, grad_b1b3, grad_b1W4, grad_b1b4],
                        [grad_W2W1, grad_W2b1, grad_W2W2, grad_W2b2, grad_W2W3, grad_W2b3, grad_W2W4, grad_W2b4],
                        [grad_b2W1, grad_b2b1, grad_b2W2, grad_b2b2, grad_b2W3, grad_b2b3, grad_b2W4, grad_b2b4],
                        [grad_W3W1, grad_W3b1, grad_W3W2, grad_W3b2, grad_W3W3, grad_W3b3, grad_W3W4, grad_W3b4],
                        [grad_b3W1, grad_b3b1, grad_b3W2, grad_b3b2, grad_b3W3, grad_b3b3, grad_b3W4, grad_b3b4],
                        [grad_W4W1, grad_W4b1, grad_W4W2, grad_W4b2, grad_W4W3, grad_W4b3, grad_W4W4, grad_W4b4], 
                        [grad_b4W1, grad_b4b1, grad_b4W2, grad_b4b2, grad_b4W3, grad_b4b3, grad_b4W4, grad_b4b4]])
    
    return Hessian



def compute_Hessian_four(opt, params, z1, z2, z3, z4, a1, a2, a3, a4, output, train_x, train_y):


    W1, b1 = params[0]                      
    W2, b2 = params[1]  
    W3, b3 = params[2]
    W4, b4 = params[3]
    W5, b5 = params[4]

    
    input_dim = opt.num_channel[0]
    m1 = opt.num_channel[1]
    m2 = opt.num_channel[2]
    m3 = opt.num_channel[3]
    m4 = opt.num_channel[4]

    
    grad_W1W1 = np.zeros((input_dim * m1, input_dim * m1))
    grad_W1b1 = np.zeros((input_dim * m1, m1))
    grad_W1W2 = np.zeros((input_dim * m1, m1*m2))
    grad_W1b2 = np.zeros((input_dim * m1, m2))
    grad_W1W3 = np.zeros((input_dim * m1, m2*m3))
    grad_W1b3 = np.zeros((input_dim * m1, m3))
    grad_W1W4 = np.zeros((input_dim * m1, m3*m4))
    grad_W1b4 = np.zeros((input_dim * m1, m4))
    grad_W1W5 = np.zeros((input_dim * m1, m4))
    grad_W1b5 = np.zeros((input_dim * m1, 1))
    
    grad_b1b1 = np.zeros((m1, m1))
    grad_b1W2 = np.zeros((m1, m1*m2))
    grad_b1b2 = np.zeros((m1, m2))
    grad_b1W3 = np.zeros((m1, m2*m3))
    grad_b1b3 = np.zeros((m1, m3))
    grad_b1W4 = np.zeros((m1, m3*m4))
    grad_b1b4 = np.zeros((m1, m4))
    grad_b1W5 = np.zeros((m1, m4))
    grad_b1b5 = np.zeros((m1, 1))

    grad_W2W2 = np.zeros((m1*m2, m1*m2))
    grad_W2b2 = np.zeros((m1*m2, m2))
    grad_W2W3 = np.zeros((m1*m2, m2*m3))
    grad_W2b3 = np.zeros((m1*m2, m3))
    grad_W2W4 = np.zeros((m1*m2, m3*m4))
    grad_W2b4 = np.zeros((m1*m2, m4))
    grad_W2W5 = np.zeros((m1*m2, m4))
    grad_W2b5 = np.zeros((m1*m2, 1))

    grad_b2b2 = np.zeros((m2, m2))
    grad_b2W3 = np.zeros((m2, m2*m3))
    grad_b2b3 = np.zeros((m2, m3))
    grad_b2W4 = np.zeros((m2, m3*m4))
    grad_b2b4 = np.zeros((m2, m4))
    grad_b2W5 = np.zeros((m2, m4))
    grad_b2b5 = np.zeros((m2, 1))

    grad_W3W3 = np.zeros((m2*m3, m2*m3))
    grad_W3b3 = np.zeros((m2*m3, m3))
    grad_W3W4 = np.zeros((m2*m3, m3*m4))
    grad_W3b4 = np.zeros((m2*m3, m4))
    grad_W3W5 = np.zeros((m2*m3, m4))
    grad_W3b5 = np.zeros((m2*m3, 1))

    grad_b3b3 = np.zeros((m3, m3))
    grad_b3W4 = np.zeros((m3, m3*m4))
    grad_b3b4 = np.zeros((m3, m4))
    grad_b3W5 = np.zeros((m3, m4))
    grad_b3b5 = np.zeros((m3, 1))

    grad_W4W4 = np.zeros((m3*m4, m3*m4))
    grad_W4b4 = np.zeros((m3*m4, m4))
    grad_W4W5 = np.zeros((m3*m4, m4))
    grad_W4b5 = np.zeros((m3*m4, 1))

    grad_b4b4 = np.zeros((m4, m4))
    grad_b4W5 = np.zeros((m4, m4))
    grad_b4b5 = np.zeros((m4, 1))

    grad_W5W5 = np.zeros((m4, m4))
    grad_W5b5 = np.zeros((m4, 1))

    q1 = 1/opt.ntrain

    print(f"q1: {q1}")

    
    for i in range(opt.ntrain):
        x_i = train_x[:,i]
        y_i = train_y[:,i]
    
        z1_i = z1[:, i]
        z1_i = z1_i.reshape(np.shape(z1_i)[0], 1)
        z2_i = z2[:, i]
        z2_i = z2_i.reshape(np.shape(z2_i)[0], 1)
        z3_i = z3[:, i]
        z3_i = z3_i.reshape(np.shape(z3_i)[0], 1)
        z4_i = z4[:, i]
        z4_i = z4_i.reshape(np.shape(z4_i)[0], 1)
        a1_i = a1[:, i]
        a1_i = a1_i.reshape(np.shape(a1_i)[0], 1)
        a2_i = a2[:, i]
        a2_i = a2_i.reshape(np.shape(a2_i)[0], 1)
        a3_i = a3[:, i]
        a3_i = a3_i.reshape(np.shape(a3_i)[0], 1)
        a4_i = a4[:, i]
        a4_i = a4_i.reshape(np.shape(a4_i)[0], 1)
        output_i = output[:, i]
        output_i = output_i.reshape(np.shape(output_i)[0], 1)
        e_i = output_i - y_i

        #------------grad_W1W1-------------------
        grad_diag_z1 = np.diag(relu_grad(z1_i.flatten()))
        grad_diag_z2 = np.diag(relu_grad(z2_i.flatten()))
        grad_diag_z3 = np.diag(relu_grad(z3_i.flatten()))
        grad_diag_z4 = np.diag(relu_grad(z4_i.flatten()))
        temp1 = (np.dot( grad_diag_z1, W2)).reshape((m1, m2))                                  #(m1, m2)  Z_1
        temp2 = (np.dot( grad_diag_z2, W3)).reshape((m2, m3))                                  #(m2, m3)  Z_2
        temp3 = (np.dot( grad_diag_z3, W4)).reshape((m3, m4))                                  #(m3, m4)  Z_3
        temp4 = (np.dot( grad_diag_z4, W5)).reshape((m4, 1))                                   #(m4, 1)   Z_4
        
        temp5 = np.dot(temp3, temp4)                                                                #(m3, 1) Z_3*Z_4
        temp6 = np.dot(temp2, temp5)                                                                #(m2, 1) Z_2*Z_3*Z4
        temp =  np.dot(temp1, temp6)                                                                 #(m1, 1) Z_1*Z_3*Z3*Z4
        temp7 = np.dot(temp1, temp2)                                                                #(m1, m2) Z_1*Z_2
        temp8 = np.dot(temp7, temp3)                                                                #(m1, m3) Z_1*Z_2*Z_3
        temp9 = np.dot(temp2, temp3)                                                                #(m2, m3) Z_2*Z_3


        #-----------
        grad_W1W1 += q1 * np.kron(np.dot(temp, temp.T), np.dot(x_i, x_i.T))                                           #(d*m1, d*m1)
        grad_W1b1 += q1 * np.kron(np.dot(temp, temp.T), x_i)                                                          #(d*m1, m1)
        grad_W1W2 += (q1 * np.dot(np.kron(temp, x_i), np.kron(temp6.T, a1_i.T)) + 
                      q1 * np.kron(np.kron(temp6.T, grad_diag_z1),  x_i) * e_i)                                        #(d*m1, m1*m2)
        grad_W1b2 += q1 * np.dot(np.kron(temp,  x_i), temp6.T)                                                         #(d*m1, m2)
        grad_W1W3 += (q1 * np.dot(np.kron(temp, x_i), np.kron(temp5.T, a2_i.T)) + 
                      q1 * np.kron(np.kron(temp5.T, np.dot(temp1, grad_diag_z2)),  x_i) * e_i)                           #(d*m1, m2*m3)
        grad_W1b3 += q1 * np.dot(np.kron(temp, x_i), temp5.T)                                                           #(d*m1, m3)
        grad_W1W4 += (q1 * np.dot(np.kron(temp, x_i), np.kron(temp4.T, a3_i.T))  + 
                      q1 * np.kron(np.kron(temp4.T,  np.dot(temp7, grad_diag_z3)), x_i) * e_i)                          #(d*m1, m3*m4)
        grad_W1b4 += q1 * np.kron(np.kron(temp, x_i), temp4.T)                                                          #(d*m1, m4)
        grad_W1W5 += (q1 * np.dot(np.kron(temp, x_i), a4_i.T)  + 
                      q1 * np.kron(np.dot(temp8, grad_diag_z4), x_i) * e_i)                                             #(d*m1, m4)
        grad_W1b5 += q1 * np.kron(temp, x_i)                                                                            #(d*m1, 1)        
        
        #-----------                                                                                        
        grad_b1b1 += q1 * np.dot(temp, temp.T)                                                                         #(m1, m1)
        grad_b1W2 += (q1 * np.dot(temp, np.kron(temp6.T, a1_i.T)) +
                      q1 * np.kron(temp6.T, grad_diag_z1) * e_i)                                                       #(m1, m1*m2) 
        grad_b1b2 += q1 * np.dot(temp, temp6.T)                                                                        #(m1, m2)
        grad_b1W3 += (q1 * np.dot(temp, np.kron(temp5.T, a2_i.T)) +
                      q1 * np.kron(temp5.T, np.dot(temp1, grad_diag_z2)) * e_i)                                        #(m1, m2*m3)
        grad_b1b3 += q1 * np.dot(temp, temp5.T)                                                                        #(m1, m3)
        grad_b1W4 += (q1 * np.dot(temp, np.kron(temp4.T, a3_i.T)) +
                      q1 * np.kron(temp4.T, np.dot(temp7, grad_diag_z3)) * e_i)                                         #(m1, m3*m4)
        grad_b1b4 += q1 * np.dot(temp, temp4.T)                                                                         #(m1, m4)
        grad_b1W5 += (q1 * np.dot(temp, a4_i.T) +
                      q1 * np.dot(temp8, grad_diag_z4) * e_i)                                                           #(m1, m4)
        grad_b1b5 += q1 * temp                                                                                          #(m1, 1)

        #-----------
        grad_W2W2 += q1 * np.dot(np.kron(temp6, a1_i),  np.kron(temp6.T, a1_i.T))                                        #(m1*m2, m1*m2)  
        grad_W2b2 += q1 * np.dot(np.kron(temp6, a1_i), temp6.T)                                                          #(m1*m2, m2)   
        grad_W2W3 += (q1 * np.dot(np.kron(temp6, a1_i), np.kron(temp5.T, a2_i.T)) +
                      q1 * np.kron(np.kron(temp5.T, grad_diag_z2), a1_i) * e_i)                                          #(m1*m2, m2*m3)   
        grad_W2b3 += q1 * np.dot(np.kron(temp6, a1_i), temp5.T)                                                          #(m1*m2, m3)  
        grad_W2W4 += (q1 * np.dot(np.kron(temp6, a1_i),  np.kron(temp4.T, a3_i.T)) +
                      q1 * np.kron(np.kron(temp4.T, np.dot(temp3, grad_diag_z3)), a1_i) * e_i)                           #(m1*m2, m3*m4)   
        grad_W2b4 += q1 * np.dot(np.kron(temp6, a1_i), temp4.T)                                                          #(m1*m2, m4)  
        grad_W2W5 += (q1 * np.dot(np.kron(temp6, a1_i),  a4_i.T) +
                      q1 * np.kron(np.dot(temp9, grad_diag_z4), a1_i) * e_i)                                     #(m1*m2, m4)   
        grad_W2b5 += q1 * np.kron(temp6, a1_i)                                                                           #(m1*m2, 1) 

        #-----------
        grad_b2b2 += q1 * np.dot(temp6, temp6.T)                                                                         #(m2, m2)   
        grad_b2W3 += (q1 * np.dot(temp6,  np.kron(temp5.T, a2_i.T)) +
                      q1 * np.kron(temp5.T, grad_diag_z2) * e_i)                                                         #(m2, m2*m3)    
        grad_b2b3 += q1 * np.dot(temp6, temp5.T)                                                                         #(m2, m3)
        grad_b2W4 += (q1 * np.dot(temp6,  np.kron(temp4.T, a3_i.T)) +
                      q1 * np.kron(temp4.T, np.dot(temp3, grad_diag_z3)) * e_i)                                          #(m2, m3*m4)    
        grad_b2b4 += q1 * np.dot(temp6, temp4.T)                                                                         #(m4, 1)
        grad_b2W5 += (q1 * np.dot(temp6,  a4_i.T) +
                      q1 * np.dot(temp9, grad_diag_z4) * e_i)                                                            #(m2, m4)    
        grad_b2b5 += q1 * temp6                                                                                          #(m4, 1)

        #-----------
        grad_W3W3 += q1 * np.dot( np.kron(temp5, a2_i), np.kron(temp5.T, a2_i.T) )                                       #(m2*m3, m2*m3)
        grad_W3b3 += q1 * np.dot( np.kron(temp5, a2_i), temp5.T)                                                         #(m2*m3, m3)    
        grad_W3W4 += (q1 * np.dot(np.kron(temp5, a2_i), np.kron(temp4.T, a3_i.T)) +
                      q1 * np.kron(np.kron(temp4.T, grad_diag_z3), a2_i) * e_i)                                            #(m2*m3, m3*m4)    
        grad_W3b4 += q1 * np.dot(np.kron(temp5, a2_i), temp4.T)                                                           #(m2*m3, m4) 
        grad_W3W5 += (q1 * np.dot(np.kron(temp5, a2_i), a4_i.T) +
                      q1 * np.kron(np.dot(temp3, grad_diag_z4), a2_i) * e_i)                                              #(m2*m3, m4)    
        grad_W3b5 += q1 * np.kron(temp5, a2_i)                                                                           #(m2*m3, 1) 


        grad_b3b3 += q1 * np.dot(temp5, temp5.T)                                                                         #(m3, m3)      
        grad_b3W4 += (q1 * np.dot(temp5, np.kron(temp4.T, a3_i.T)) +
                      q1 * np.kron(temp4.T, grad_diag_z3) * e_i)                                                         #(m3, m3*m4)    
        grad_b3b4 += q1 * np.dot(temp5, temp4.T)                                                                         #(m3, m4)
        grad_b3W5 += (q1 * np.dot(temp5, a4_i.T) +
                      q1 * np.dot(temp3, grad_diag_z4) * e_i)                                                            #(m3, m4)    
        grad_b3b5 += q1 *  temp5                                                                                         #(m3, 1)
        
        grad_W4W4 += q1 * np.kron(np.dot(temp4, temp4.T), np.dot(a3_i, a3_i.T))                                          #(m3*m4, m3*m4)    
        grad_W4b4 += q1 * np.kron(np.dot(temp4, temp4.T), a3_i)                                                          #(m3*m4, m4)
        grad_W4W5 += (q1 * np.dot(np.kron(temp4, a3_i), a4_i.T)                                                          #(m3*m4, m4)    
                      + q1 * np.kron(grad_diag_z4, a3_i) * e_i) 
        grad_W4b5 += q1 * np.kron(temp4, a3_i)                                                                         #(m3*m4, 1)


        grad_b4b4 += q1 * np.dot(temp4, temp4.T)                                                                         #(m4, m4)
        grad_b4W5 += (q1 * np.dot(temp4, a4_i.T) +                                                                       #(m4, m4)   
                      q1 * grad_diag_z4 * e_i) 
        grad_b4b5 += q1 * temp4                                                                                          #(m4, 1)    

        grad_W5W5 += q1 * np.dot(a4_i, a4_i.T)                                                                           #(m4, m4)
        grad_W5b5 += q1 * a4_i                                                                                           #(m4, 1)


    
        
    grad_b1W1 = grad_W1b1.T                                                                                 
    
    grad_W2W1 = grad_W1W2.T                                                                                    
    grad_W2b1 = grad_b1W2.T                                                                                 

    grad_b2W1 = grad_W1b2.T                                                                                   
    grad_b2b1 = grad_b1b2.T                                                                                    
    grad_b2W2 = grad_W2b2.T                                                                                    

    grad_W3W1 = grad_W1W3.T                                                                                    
    grad_W3b1 = grad_b1W3.T                                                                                    
    grad_W3W2 = grad_W2W3.T                                                                                    
    grad_W3b2 = grad_b2W3.T                                                                                   

    grad_b3W1 = grad_W1b3.T                                                                                   
    grad_b3b1 = grad_b1b3.T                                                                                  
    grad_b3W2 = grad_W2b3.T                                                                                    
    grad_b3b2 = grad_b2b3.T                                                                                  
    grad_b3W3 = grad_W3b3.T                                                                                    

    grad_W4W1 = grad_W1W4.T                                                                                    
    grad_W4b1 = grad_b1W4.T                                                                                    
    grad_W4W2 = grad_W2W4.T                                                                                    
    grad_W4b2 = grad_b2W4.T                                                                                    
    grad_W4W3 = grad_W3W4.T                                                                                      
    grad_W4b3 = grad_b3W4.T      

    grad_b4W1 = grad_W1b4.T                                                                                    
    grad_b4b1 = grad_b1b4.T                                                                                   
    grad_b4W2 = grad_W2b4.T                                                                                    
    grad_b4b2 = grad_b2b4.T                                                                                    
    grad_b4W3 = grad_W3b4.T                                                                                    
    grad_b4b3 = grad_b3b4.T    
    grad_b4W4 = grad_W4b4.T    

    grad_W5W1 = grad_W1W5.T                                                                                    
    grad_W5b1 = grad_b1W5.T                                                                                   
    grad_W5W2 = grad_W2W5.T                                                                                    
    grad_W5b2 = grad_b2W5.T                                                                                    
    grad_W5W3 = grad_W3W5.T                                                                                    
    grad_W5b3 = grad_b3W5.T    
    grad_W5W4 = grad_W4W5.T 
    grad_W5b4 = grad_b4W5.T  

    grad_b5W1 = grad_W1b5.T                                                                                    
    grad_b5b1 = grad_b1b5.T                                                                                   
    grad_b5W2 = grad_W2b5.T                                                                                    
    grad_b5b2 = grad_b2b5.T                                                                                    
    grad_b5W3 = grad_W3b5.T                                                                                    
    grad_b5b3 = grad_b3b5.T    
    grad_b5W4 = grad_W4b5.T  
    grad_b5b4 = grad_b4b5.T 
    grad_b5W5 = grad_W5b5.T 
    grad_b5b5 = 1

    
    
    Hessian = np.block([[grad_W1W1, grad_W1b1, grad_W1W2, grad_W1b2, grad_W1W3, grad_W1b3, grad_W1W4, grad_W1b4, grad_W1W5, grad_W1b5],
                        [grad_b1W1, grad_b1b1, grad_b1W2, grad_b1b2, grad_b1W3, grad_b1b3, grad_b1W4, grad_b1b4, grad_b1W5, grad_b1b5],
                        [grad_W2W1, grad_W2b1, grad_W2W2, grad_W2b2, grad_W2W3, grad_W2b3, grad_W2W4, grad_W2b4, grad_W2W5, grad_W2b5],
                        [grad_b2W1, grad_b2b1, grad_b2W2, grad_b2b2, grad_b2W3, grad_b2b3, grad_b2W4, grad_b2b4, grad_b2W5, grad_b2b5],
                        [grad_W3W1, grad_W3b1, grad_W3W2, grad_W3b2, grad_W3W3, grad_W3b3, grad_W3W4, grad_W3b4, grad_W3W5, grad_W3b5],
                        [grad_b3W1, grad_b3b1, grad_b3W2, grad_b3b2, grad_b3W3, grad_b3b3, grad_b3W4, grad_b3b4, grad_b3W5, grad_b3b5],
                        [grad_W4W1, grad_W4b1, grad_W4W2, grad_W4b2, grad_W4W3, grad_W4b3, grad_W4W4, grad_W4b4, grad_W4W5, grad_W4b5], 
                        [grad_b4W1, grad_b4b1, grad_b4W2, grad_b4b2, grad_b4W3, grad_b4b3, grad_b4W4, grad_b4b4, grad_b4W5, grad_b4b5],
                        [grad_W5W1, grad_W5b1, grad_W5W2, grad_W5b2, grad_W5W3, grad_W5b3, grad_W5W4, grad_W5b4, grad_W5W5, grad_W5b5], 
                        [grad_b5W1, grad_b5b1, grad_b5W2, grad_b5b2, grad_b5W3, grad_b5b3, grad_b5W4, grad_b5b4, grad_b5W5, grad_b5b5]])
    
    return Hessian









