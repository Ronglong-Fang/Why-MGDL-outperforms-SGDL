import jax.numpy as np


def relu_grad(x):
    return np.where(x >= 0, 1, 0)

def compute_Hessian_layer4_scale4(opt, params, z, a, output, train_x, train_y):




    input_dim = opt.num_channel[0]
    m1 = opt.num_channel[1]
    m2 = opt.num_channel[2]
    m3 = opt.num_channel[3]
    m4 = opt.num_channel[4]
    
    grad = {}
    q1 = 1/opt.ntrain
    Hessian = {}
    
    for k in range(1, opt.scale+1):
        
        for j in range(k, opt.scale+1):
            
            W1k, b1k = params['scale'+str(k)][0]                      
            W2k, b2k = params['scale'+str(k)][1]  
            W3k, b3k = params['scale'+str(k)][2]
            W4k, b4k = params['scale'+str(k)][3]
            W5k, b5k = params['scale'+str(k)][4]

            W1j, b1j = params['scale'+str(j)][0]                      
            W2j, b2j = params['scale'+str(j)][1]  
            W3j, b3j = params['scale'+str(j)][2]
            W4j, b4j = params['scale'+str(j)][3]
            W5j, b5j = params['scale'+str(j)][4]


            
            grad_W1k_W1j = np.zeros((input_dim * m1, input_dim * m1))
            grad_W1k_b1j = np.zeros((input_dim * m1, m1))
            grad_W1k_W2j = np.zeros((input_dim * m1, m1*m2))
            grad_W1k_b2j = np.zeros((input_dim * m1, m2))
            grad_W1k_W3j = np.zeros((input_dim * m1, m2*m3))
            grad_W1k_b3j = np.zeros((input_dim * m1, m3))
            grad_W1k_W4j = np.zeros((input_dim * m1, m3*m4))
            grad_W1k_b4j = np.zeros((input_dim * m1, m4))
            grad_W1k_W5j = np.zeros((input_dim * m1, m4))
            grad_W1k_b5j = np.zeros((input_dim * m1, 1))

            grad_b1k_W1j = np.zeros((m1, input_dim * m1))
            grad_b1k_b1j = np.zeros((m1, m1))
            grad_b1k_W2j = np.zeros((m1, m1*m2))
            grad_b1k_b2j = np.zeros((m1, m2))
            grad_b1k_W3j = np.zeros((m1, m2*m3))
            grad_b1k_b3j = np.zeros((m1, m3))
            grad_b1k_W4j = np.zeros((m1, m3*m4))
            grad_b1k_b4j = np.zeros((m1, m4))
            grad_b1k_W5j = np.zeros((m1, m4))
            grad_b1k_b5j = np.zeros((m1, 1))

            grad_W2k_W1j = np.zeros((m1*m2, input_dim * m1))
            grad_W2k_b1j = np.zeros((m1*m2, m1))
            grad_W2k_W2j = np.zeros((m1*m2, m1*m2))
            grad_W2k_b2j = np.zeros((m1*m2, m2))
            grad_W2k_W3j = np.zeros((m1*m2, m2*m3))
            grad_W2k_b3j = np.zeros((m1*m2, m3))
            grad_W2k_W4j = np.zeros((m1*m2, m3*m4))
            grad_W2k_b4j = np.zeros((m1*m2, m4))
            grad_W2k_W5j = np.zeros((m1*m2, m4))
            grad_W2k_b5j = np.zeros((m1*m2, 1))

            grad_b2k_W1j = np.zeros((m2, input_dim * m1))
            grad_b2k_b1j = np.zeros((m2, m1))
            grad_b2k_W2j = np.zeros((m2, m1*m2))
            grad_b2k_b2j = np.zeros((m2, m2))
            grad_b2k_W3j = np.zeros((m2, m2*m3))
            grad_b2k_b3j = np.zeros((m2, m3))
            grad_b2k_W4j = np.zeros((m2, m3*m4))
            grad_b2k_b4j = np.zeros((m2, m4))
            grad_b2k_W5j = np.zeros((m2, m4))
            grad_b2k_b5j = np.zeros((m2, 1))

            grad_W3k_W1j = np.zeros((m2*m3, input_dim * m1))
            grad_W3k_b1j = np.zeros((m2*m3, m1))
            grad_W3k_W2j = np.zeros((m2*m3, m1*m2))
            grad_W3k_b2j = np.zeros((m2*m3, m2))
            grad_W3k_W3j = np.zeros((m2*m3, m2*m3))
            grad_W3k_b3j = np.zeros((m2*m3, m3))
            grad_W3k_W4j = np.zeros((m2*m3, m3*m4))
            grad_W3k_b4j = np.zeros((m2*m3, m4))
            grad_W3k_W5j = np.zeros((m2*m3, m4))
            grad_W3k_b5j = np.zeros((m2*m3, 1))

            grad_b3k_W1j = np.zeros((m3, input_dim * m1))
            grad_b3k_b1j = np.zeros((m3, m1))
            grad_b3k_W2j = np.zeros((m3, m1*m2))
            grad_b3k_b2j = np.zeros((m3, m2))
            grad_b3k_W3j = np.zeros((m3, m2*m3))
            grad_b3k_b3j = np.zeros((m3, m3))
            grad_b3k_W4j = np.zeros((m3, m3*m4))
            grad_b3k_b4j = np.zeros((m3, m4))
            grad_b3k_W5j = np.zeros((m3, m4))
            grad_b3k_b5j = np.zeros((m3, 1))

            grad_W4k_W1j = np.zeros((m3*m4, input_dim * m1))
            grad_W4k_b1j = np.zeros((m3*m4, m1))
            grad_W4k_W2j = np.zeros((m3*m4, m1*m2))
            grad_W4k_b2j = np.zeros((m3*m4, m2))
            grad_W4k_W3j = np.zeros((m3*m4, m2*m3))
            grad_W4k_b3j = np.zeros((m3*m4, m3))
            grad_W4k_W4j = np.zeros((m3*m4, m3*m4))
            grad_W4k_b4j = np.zeros((m3*m4, m4))
            grad_W4k_W5j = np.zeros((m3*m4, m4))
            grad_W4k_b5j = np.zeros((m3*m4, 1))

            grad_b4k_W1j = np.zeros((m4, input_dim * m1))
            grad_b4k_b1j = np.zeros((m4, m1))
            grad_b4k_W2j = np.zeros((m4, m1*m2))
            grad_b4k_b2j = np.zeros((m4, m2))
            grad_b4k_W3j = np.zeros((m4, m2*m3))
            grad_b4k_b3j = np.zeros((m4, m3))
            grad_b4k_W4j = np.zeros((m4, m3*m4))
            grad_b4k_b4j = np.zeros((m4, m4))
            grad_b4k_W5j = np.zeros((m4, m4))
            grad_b4k_b5j = np.zeros((m4, 1))

            grad_W5k_W1j = np.zeros((m4, input_dim * m1))
            grad_W5k_b1j = np.zeros((m4, m1))
            grad_W5k_W2j = np.zeros((m4, m1*m2))
            grad_W5k_b2j = np.zeros((m4, m2))
            grad_W5k_W3j = np.zeros((m4, m2*m3))
            grad_W5k_b3j = np.zeros((m4, m3))
            grad_W5k_W4j = np.zeros((m4, m3*m4))
            grad_W5k_b4j = np.zeros((m4, m4))
            grad_W5k_W5j = np.zeros((m4, m4))
            grad_W5k_b5j = np.zeros((m4, 1))

            grad_b5k_W1j = np.zeros((1, input_dim * m1))
            grad_b5k_b1j = np.zeros((1, m1))
            grad_b5k_W2j = np.zeros((1, m1*m2))
            grad_b5k_b2j = np.zeros((1, m2))
            grad_b5k_W3j = np.zeros((1, m2*m3))
            grad_b5k_b3j = np.zeros((1, m3))
            grad_b5k_W4j = np.zeros((1, m3*m4))
            grad_b5k_b4j = np.zeros((1, m4))
            grad_b5k_W5j = np.zeros((1, m4))
            grad_b5k_b5j = np.zeros((1, 1))
        

            if k==j:  
                
                for i in range(opt.ntrain):

                    #----------
                    x_i = train_x[:,i]
                    x_i = opt.coeff['scale'+str(k)] * x_i.reshape(input_dim, 1)
                    y_i = train_y[:,i]

                    #----------
                    z1_i = z[1]['scale'+str(k)][:, i]
                    z1_i = z1_i.reshape(np.shape(z1_i)[0], 1)
                    
                    z2_i = z[2]['scale'+str(k)][:, i]
                    z2_i = z2_i.reshape(np.shape(z2_i)[0], 1)
                    
                    z3_i = z[3]['scale'+str(k)][:, i]
                    z3_i = z3_i.reshape(np.shape(z3_i)[0], 1)
                    
                    z4_i = z[4]['scale'+str(k)][:, i]
                    z4_i = z4_i.reshape(np.shape(z4_i)[0], 1)

                    
                    #----------
                    a1_i = a[1]['scale'+str(k)][:, i]
                    a1_i = a1_i.reshape(np.shape(a1_i)[0], 1)
                    
                    a2_i = a[2]['scale'+str(k)][:, i]
                    a2_i = a2_i.reshape(np.shape(a2_i)[0], 1)
                    
                    a3_i = a[3]['scale'+str(k)][:, i]
                    a3_i = a3_i.reshape(np.shape(a3_i)[0], 1)
                    
                    a4_i = a[4]['scale'+str(k)][:, i]
                    a4_i = a4_i.reshape(np.shape(a4_i)[0], 1)

                    #----------
                    output_i = output[:, i]
                    output_i = output_i.reshape(np.shape(output_i)[0], 1)
                    e_i = output_i - y_i
            
                    #------------grad_W1W1-------------------
                    grad_diag_z1 = np.diag(relu_grad(z1_i.flatten()))
                    grad_diag_z2 = np.diag(relu_grad(z2_i.flatten()))
                    grad_diag_z3 = np.diag(relu_grad(z3_i.flatten()))
                    grad_diag_z4 = np.diag(relu_grad(z4_i.flatten()))
                    temp1 = (np.dot( grad_diag_z1, W2k)).reshape((m1, m2))                                  #(m1, m2)  Z_1
                    temp2 = (np.dot( grad_diag_z2, W3k)).reshape((m2, m3))                                  #(m2, m3)  Z_2
                    temp3 = (np.dot( grad_diag_z3, W4k)).reshape((m3, m4))                                  #(m3, m4)  Z_3
                    temp4 = (np.dot( grad_diag_z4, W5k)).reshape((m4, 1))                                   #(m4, 1)   Z_4
                    
                    temp5 = np.dot(temp3, temp4)                                                                #(m3, 1) Z_3*Z_4
                    temp6 = np.dot(temp2, temp5)                                                                #(m2, 1) Z_2*Z_3*Z4
                    temp = np.dot(temp1, temp6)                                                                 #(m1, 1) Z_1*Z_3*Z3*Z4
                    temp7 = np.dot(temp1, temp2)                                                                #(m1, m2) Z_1*Z_2
                    temp8 = np.dot(temp7, temp3)                                                                #(m1, m3) Z_1*Z_2*Z_3
                    temp9 = np.dot(temp2, temp3)                                                                #(m2, m3) Z_2*Z_3
            
            
                    #-----------
                    grad_W1k_W1j += q1 * np.kron(np.dot(temp, temp.T), np.dot(x_i, x_i.T))                                           #(d*m1, d*m1)
                    grad_W1k_b1j += q1 * np.kron(np.dot(temp, temp.T), x_i)                                                          #(d*m1, m1)
                    grad_W1k_W2j += q1 * (np.dot(np.kron(temp, x_i), np.kron(temp6.T, a1_i.T)) + 
                                          np.kron(np.kron(temp6.T, grad_diag_z1),  x_i) * e_i)                                        #(d*m1, m1*m2)
                    grad_W1k_b2j += q1 * np.dot(np.kron(temp,  x_i), temp6.T)                                                         #(d*m1, m2)
                    grad_W1k_W3j += q1 * (np.dot(np.kron(temp, x_i), np.kron(temp5.T, a2_i.T)) +
                                          np.kron(np.kron(temp5.T, np.dot(temp1, grad_diag_z2)),  x_i) * e_i)                          #(d*m1, m2*m3)
                    grad_W1k_b3j += q1 * np.dot(np.kron(temp, x_i), temp5.T)                                                           #(d*m1, m3)
                    grad_W1k_W4j += q1 * (np.dot(np.kron(temp, x_i), np.kron(temp4.T, a3_i.T))  + 
                                          np.kron(np.kron(temp4.T,  np.dot(temp7, grad_diag_z3)), x_i) * e_i)                          #(d*m1, m3*m4)
                    grad_W1k_b4j += q1 * np.dot(np.kron(temp, x_i), temp4.T)                                                          #(d*m1, m4)
                    grad_W1k_W5j += q1 * (np.dot(np.kron(temp, x_i), a4_i.T)  + 
                                          np.kron(np.dot(temp8, grad_diag_z4), x_i) * e_i)                                             #(d*m1, m4)
                    grad_W1k_b5j += q1 * np.kron(temp, x_i)                                                                            #(d*m1, 1)        
                    
                    #-----------                                                                                        
                    grad_b1k_b1j += q1 * np.dot(temp, temp.T)                                                                          #(m1, m1)
                    grad_b1k_W2j += q1 * (np.dot(temp, np.kron(temp6.T, a1_i.T)) +
                                          np.kron(temp6.T, grad_diag_z1) * e_i)                                                        #(m1, m1*m2) 
                    grad_b1k_b2j += q1 * np.dot(temp, temp6.T)                                                                         #(m1, m2)
                    grad_b1k_W3j += q1 * (np.dot(temp, np.kron(temp5.T, a2_i.T)) +
                                          np.kron(temp5.T, np.dot(temp1, grad_diag_z2)) * e_i)                                         #(m1, m2*m3)
                    grad_b1k_b3j += q1 * np.dot(temp, temp5.T)                                                                         #(m1, m3)
                    grad_b1k_W4j += q1 * (np.dot(temp, np.kron(temp4.T, a3_i.T)) +
                                          np.kron(temp4.T, np.dot(temp7, grad_diag_z3)) * e_i)                                         #(m1, m3*m4)
                    grad_b1k_b4j += q1 * np.dot(temp, temp4.T)                                                                         #(m1, m4)
                    grad_b1k_W5j += q1 * (np.dot(temp, a4_i.T) +
                                          np.dot(temp8, grad_diag_z4) * e_i)                                                           #(m1, m4)
                    grad_b1k_b5j += q1 * temp                                                                                          #(m1, 1)
            
                    #-----------
                    grad_W2k_W2j += q1 * np.dot(np.kron(temp6, a1_i),  np.kron(temp6.T, a1_i.T))                                        #(m1*m2, m1*m2)  
                    grad_W2k_b2j += q1 * np.dot(np.kron(temp6, a1_i), temp6.T)                                                          #(m1*m2, m2)   
                    grad_W2k_W3j += q1 * (np.dot(np.kron(temp6, a1_i), np.kron(temp5.T, a2_i.T)) +
                                          np.kron(np.kron(temp5.T, grad_diag_z2), a1_i) * e_i)                                          #(m1*m2, m2*m3)   
                    grad_W2k_b3j += q1 * np.dot(np.kron(temp6, a1_i), temp5.T)                                                          #(m1*m2, m3)  
                    grad_W2k_W4j += q1 * (np.dot(np.kron(temp6, a1_i),  np.kron(temp4.T, a3_i.T)) +
                                          np.kron(np.kron(temp4.T, np.dot(temp3, grad_diag_z3)), a1_i) * e_i)                             #(m1*m2, m3*m4)   
                    grad_W2k_b4j += q1 * np.dot(np.kron(temp6, a1_i), temp4.T)                                                          #(m1*m2, m4)  
                    grad_W2k_W5j += q1 * (np.dot(np.kron(temp6, a1_i),  a4_i.T) +
                                          np.kron(np.dot(temp9, grad_diag_z4), a1_i) * e_i)                                                #(m1*m2, m4)   
                    grad_W2k_b5j += q1 * np.kron(temp6, a1_i)                                                                           #(m1*m2, 1) 
            
                    #-----------
                    grad_b2k_b2j += q1 * np.dot(temp6, temp6.T)                                                                         #(m2, m2)   
                    grad_b2k_W3j += q1 * (np.dot(temp6,  np.kron(temp5.T, a2_i.T)) +
                                          np.kron(temp5.T, grad_diag_z2) * e_i)                                                         #(m2, m2*m3)    
                    grad_b2k_b3j += q1 * np.dot(temp6, temp5.T)                                                                         #(m2, m3)
                    grad_b2k_W4j += q1 * (np.dot(temp6,  np.kron(temp4.T, a3_i.T)) +
                                          np.kron(temp4.T, np.dot(temp3, grad_diag_z3)) * e_i)                                          #(m2, m3*m4)    
                    grad_b2k_b4j += q1 * np.dot(temp6, temp4.T)                                                                         #(m4, 1)
                    grad_b2k_W5j += q1 * (np.dot(temp6,  a4_i.T) +
                                          np.dot(temp9, grad_diag_z4) * e_i)                                                            #(m2, m4)    
                    grad_b2k_b5j += q1 * temp6                                                                                          #(m4, 1)
            
                    #-----------
                    grad_W3k_W3j += q1 * np.dot(np.kron(temp5, a2_i), np.kron(temp5.T, a2_i.T) )                                       #(m2*m3, m2*m3)
                    grad_W3k_b3j += q1 * np.dot(np.kron(temp5, a2_i), temp5.T)                                                         #(m2*m3, m3)    
                    grad_W3k_W4j += q1 * (np.dot(np.kron(temp5, a2_i), np.kron(temp4.T, a3_i.T)) +
                                          np.kron(np.kron(temp4.T, grad_diag_z3), a2_i) * e_i)                                            #(m2*m3, m3*m4)    
                    grad_W3k_b4j += q1 * np.dot(np.kron(temp5, a2_i), temp4.T)                                                           #(m2*m3, m4) 
                    grad_W3k_W5j += q1 * (np.dot(np.kron(temp5, a2_i), a4_i.T) +
                                          np.kron(np.dot(temp3, grad_diag_z4), a2_i) * e_i)                                              #(m2*m3, m4)    
                    grad_W3k_b5j += q1 * np.kron(temp5, a2_i)                                                                           #(m2*m3, 1) 
            
                    #-------------
                    grad_b3k_b3j += q1 * np.dot(temp5, temp5.T)                                                                         #(m3, m3)      
                    grad_b3k_W4j += q1 * (np.dot(temp5, np.kron(temp4.T, a3_i.T)) +
                                          np.kron(temp4.T, grad_diag_z3) * e_i)                                                         #(m3, m3*m4)    
                    grad_b3k_b4j += q1 * np.dot(temp5, temp4.T)                                                                         #(m3, m4)
                    grad_b3k_W5j += q1 * (np.dot(temp5, a4_i.T) +
                                          np.dot(temp3, grad_diag_z4) * e_i)                                                            #(m3, m4)    
                    grad_b3k_b5j += q1 *  temp5                                                                                         #(m3, 1)

                    #-------------
                    grad_W4k_W4j += q1 * np.kron(np.dot(temp4, temp4.T), np.dot(a3_i, a3_i.T))                                          #(m3*m4, m3*m4)    
                    grad_W4k_b4j += q1 * np.kron(np.dot(temp4, temp4.T), a3_i)                                                          #(m3*m4, m4)
                    grad_W4k_W5j += q1 * (np.dot(np.kron(temp4, a3_i), a4_i.T) + 
                                          np.kron(grad_diag_z4, a3_i) * e_i) 
                    grad_W4k_b5j += q1 * np.kron(temp4, a3_i)                                                                         #(m3*m4, 1)
            
                    #--------------
                    grad_b4k_b4j += q1 * np.dot(temp4, temp4.T)                                                                         #(m4, m4)
                    grad_b4k_W5j += q1 * (np.dot(temp4, a4_i.T) + 
                                          grad_diag_z4 * e_i) 
                    grad_b4k_b5j += q1 * temp4                                                                                          #(m4, 1)    

                    #--------------
                    grad_W5k_W5j += q1 * np.dot(a4_i, a4_i.T)                                                                           #(m4, m4)
                    grad_W5k_b5j += q1 * a4_i                                                                                           #(m4, 1)
            
            
                
                    
                grad_b1k_W1j = grad_W1k_b1j.T                                                                                 
                
                grad_W2k_W1j = grad_W1k_W2j.T                                                                                    
                grad_W2k_b1j = grad_b1k_W2j.T                                                                                 
            
                grad_b2k_W1j = grad_W1k_b2j.T                                                                                   
                grad_b2k_b1j = grad_b1k_b2j.T                                                                                    
                grad_b2k_W2j = grad_W2k_b2j.T                                                                                    
            
                grad_W3k_W1j = grad_W1k_W3j.T                                                                                    
                grad_W3k_b1j = grad_b1k_W3j.T                                                                                    
                grad_W3k_W2j = grad_W2k_W3j.T                                                                                    
                grad_W3k_b2j = grad_b2k_W3j.T                                                                                   
            
                grad_b3k_W1j = grad_W1k_b3j.T                                                                                   
                grad_b3k_b1j = grad_b1k_b3j.T                                                                                  
                grad_b3k_W2j = grad_W2k_b3j.T                                                                                    
                grad_b3k_b2j = grad_b2k_b3j.T                                                                                  
                grad_b3k_W3j = grad_W3k_b3j.T                                                                                    
            
                grad_W4k_W1j = grad_W1k_W4j.T                                                                                    
                grad_W4k_b1j = grad_b1k_W4j.T                                                                                    
                grad_W4k_W2j = grad_W2k_W4j.T                                                                                    
                grad_W4k_b2j = grad_b2k_W4j.T                                                                                    
                grad_W4k_W3j = grad_W3k_W4j.T                                                                                      
                grad_W4k_b3j = grad_b3k_W4j.T      
            
                grad_b4k_W1j = grad_W1k_b4j.T                                                                                    
                grad_b4k_b1j = grad_b1k_b4j.T                                                                                   
                grad_b4k_W2j = grad_W2k_b4j.T                                                                                    
                grad_b4k_b2j = grad_b2k_b4j.T                                                                                    
                grad_b4k_W3j = grad_W3k_b4j.T                                                                                    
                grad_b4k_b3j = grad_b3k_b4j.T    
                grad_b4k_W4j = grad_W4k_b4j.T    
            
                grad_W5k_W1j = grad_W1k_W5j.T                                                                                    
                grad_W5k_b1j = grad_b1k_W5j.T                                                                                   
                grad_W5k_W2j = grad_W2k_W5j.T                                                                                    
                grad_W5k_b2j = grad_b2k_W5j.T                                                                                    
                grad_W5k_W3j = grad_W3k_W5j.T                                                                                    
                grad_W5k_b3j = grad_b3k_W5j.T    
                grad_W5k_W4j = grad_W4k_W5j.T 
                grad_W5k_b4j = grad_b4k_W5j.T  
            
                grad_b5k_W1j = grad_W1k_b5j.T                                                                                    
                grad_b5k_b1j = grad_b1k_b5j.T                                                                                   
                grad_b5k_W2j = grad_W2k_b5j.T                                                                                    
                grad_b5k_b2j = grad_b2k_b5j.T                                                                                    
                grad_b5k_W3j = grad_W3k_b5j.T                                                                                    
                grad_b5k_b3j = grad_b3k_b5j.T    
                grad_b5k_W4j = grad_W4k_b5j.T  
                grad_b5k_b4j = grad_b4k_b5j.T 
                grad_b5k_W5j = grad_W5k_b5j.T 
                grad_b5k_b5j = 1


            else:
                
                for i in range(opt.ntrain):
                    #----------
                    x_i = train_x[:,i]
                    x_ik = opt.coeff['scale'+str(k)] * x_i.reshape(input_dim, 1)
                    x_ij = opt.coeff['scale'+str(j)] * x_i.reshape(input_dim, 1)
                    
                    y_i = train_y[:,i]

                    #----------k scale---------------------------
                    z1_ik = z[1]['scale'+str(k)][:, i]
                    z1_ik = z1_ik.reshape(np.shape(z1_ik)[0], 1)
                    
                    z2_ik = z[2]['scale'+str(k)][:, i]
                    z2_ik = z2_ik.reshape(np.shape(z2_ik)[0], 1)
                    
                    z3_ik = z[3]['scale'+str(k)][:, i]
                    z3_ik = z3_ik.reshape(np.shape(z3_ik)[0], 1)
                    
                    z4_ik = z[4]['scale'+str(k)][:, i]
                    z4_ik = z4_ik.reshape(np.shape(z4_ik)[0], 1)
                    
                    #----------j scale---------------------------
                    z1_ij = z[1]['scale'+str(j)][:, i]
                    z1_ij = z1_ij.reshape(np.shape(z1_ij)[0], 1)
                    
                    z2_ij = z[2]['scale'+str(j)][:, i]
                    z2_ij = z2_ij.reshape(np.shape(z2_ij)[0], 1)
                    
                    z3_ij = z[3]['scale'+str(j)][:, i]
                    z3_ij = z3_ij.reshape(np.shape(z3_ij)[0], 1)
                    
                    z4_ij = z[4]['scale'+str(j)][:, i]
                    z4_ij = z4_ij.reshape(np.shape(z4_ij)[0], 1)

                    
                    #----------k scale---------------------------
                    a1_ik = a[1]['scale'+str(k)][:, i]
                    a1_ik = a1_ik.reshape(np.shape(a1_ik)[0], 1)
                    
                    a2_ik = a[2]['scale'+str(k)][:, i]
                    a2_ik = a2_ik.reshape(np.shape(a2_ik)[0], 1)
                    
                    a3_ik = a[3]['scale'+str(k)][:, i]
                    a3_ik = a3_ik.reshape(np.shape(a3_ik)[0], 1)
                    
                    a4_ik = a[4]['scale'+str(k)][:, i]
                    a4_ik = a4_ik.reshape(np.shape(a4_ik)[0], 1)

                    #----------j scale--------------------------
                    a1_ij = a[1]['scale'+str(j)][:, i]
                    a1_ij = a1_ij.reshape(np.shape(a1_ij)[0], 1)
                    
                    a2_ij = a[2]['scale'+str(j)][:, i]
                    a2_ij = a2_ij.reshape(np.shape(a2_ij)[0], 1)
                    
                    a3_ij = a[3]['scale'+str(j)][:, i]
                    a3_ij = a3_ij.reshape(np.shape(a3_ij)[0], 1)
                    
                    a4_ij = a[4]['scale'+str(j)][:, i]
                    a4_ij = a4_ij.reshape(np.shape(a4_ij)[0], 1)

                    #---------------------------------------------
                    output_i = output[:, i]
                    output_i = output_i.reshape(np.shape(output_i)[0], 1)
                    e_i = output_i - y_i
            
                    grad_diag_z1k = np.diag(relu_grad(z1_ik.flatten()))
                    grad_diag_z2k = np.diag(relu_grad(z2_ik.flatten()))
                    grad_diag_z3k = np.diag(relu_grad(z3_ik.flatten()))
                    grad_diag_z4k = np.diag(relu_grad(z4_ik.flatten()))
                    temp1k = (np.dot( grad_diag_z1k, W2k)).reshape((m1, m2))                                  #(m1, m2)  Z_1
                    temp2k = (np.dot( grad_diag_z2k, W3k)).reshape((m2, m3))                                  #(m2, m3)  Z_2
                    temp3k = (np.dot( grad_diag_z3k, W4k)).reshape((m3, m4))                                  #(m3, m4)  Z_3
                    temp4k = (np.dot( grad_diag_z4k, W5k)).reshape((m4, 1))                                   #(m4, 1)   Z_4
                    
                    temp5k = np.dot(temp3k, temp4k)                                                                #(m3, 1) Z_3*Z_4
                    temp6k = np.dot(temp2k, temp5k)                                                                #(m2, 1) Z_2*Z_3*Z4
                    tempk = np.dot(temp1k, temp6k)                                                                 #(m1, 1) Z_1*Z_3*Z3*Z4



                    grad_diag_z1j = np.diag(relu_grad(z1_ij.flatten()))
                    grad_diag_z2j = np.diag(relu_grad(z2_ij.flatten()))
                    grad_diag_z3j = np.diag(relu_grad(z3_ij.flatten()))
                    grad_diag_z4j = np.diag(relu_grad(z4_ij.flatten()))
                    temp1j = (np.dot( grad_diag_z1j, W2j)).reshape((m1, m2))                                  #(m1, m2)  Z_1
                    temp2j = (np.dot( grad_diag_z2j, W3j)).reshape((m2, m3))                                  #(m2, m3)  Z_2
                    temp3j = (np.dot( grad_diag_z3j, W4j)).reshape((m3, m4))                                  #(m3, m4)  Z_3
                    temp4j = (np.dot( grad_diag_z4j, W5j)).reshape((m4, 1))                                   #(m4, 1)   Z_4
                    
                    temp5j = np.dot(temp3j, temp4j)                                                                #(m3, 1) Z_3*Z_4
                    temp6j = np.dot(temp2j, temp5j)                                                                #(m2, 1) Z_2*Z_3*Z4
                    tempj = np.dot(temp1j, temp6j)                                                                 #(m1, 1) Z_1*Z_3*Z3*Z4



                    #-----------
                    grad_W1k_W1j += q1 * np.dot(np.kron(tempk, x_ik), np.kron(tempj.T, x_ij.T))                                          
                    grad_W1k_b1j += q1 * np.dot(np.kron(tempk, x_ik), tempj.T)                                                             
                    grad_W1k_W2j += q1 * np.dot(np.kron(tempk, x_ik), np.kron(temp6j.T, a1_ij.T))                                          
                    grad_W1k_b2j += q1 * np.dot(np.kron(tempk, x_ik), temp6j.T)                                                           
                    grad_W1k_W3j += q1 * np.dot(np.kron(tempk, x_ik), np.kron(temp5j.T, a2_ij.T))                                          
                    grad_W1k_b3j += q1 * np.dot(np.kron(tempk, x_ik), temp5j.T)                                                            
                    grad_W1k_W4j += q1 * np.dot(np.kron(tempk, x_ik), np.kron(temp4j.T, a3_ij.T))                                         
                    grad_W1k_b4j += q1 * np.dot(np.kron(tempk, x_ik), temp4j.T)                                                            
                    grad_W1k_W5j += q1 * np.dot(np.kron(tempk, x_ik), a4_ij.T)                                                            
                    grad_W1k_b5j += q1 * np.kron(tempk, x_ik)                                                                              

                    #-------------
                    grad_b1k_W1j += q1 * np.dot(tempk, np.kron(tempj.T, x_ij.T)) 
                    grad_b1k_b1j += q1 * np.dot(tempk, tempj.T)                                                                        
                    grad_b1k_W2j += q1 * np.dot(tempk, np.kron(temp6j.T, a1_ij.T))
                    grad_b1k_W3j += q1 * np.dot(tempk, np.kron(temp5j.T, a2_ij.T))
                    grad_b1k_b3j += q1 * np.dot(tempk, temp5j.T)                                                                     
                    grad_b1k_W4j += q1 * np.dot(tempk, np.kron(temp4j.T, a3_ij.T)) 
                    grad_b1k_b4j += q1 * np.dot(tempk, temp4j.T)                                                                      
                    grad_b1k_W5j += q1 * np.dot(tempk, a4_ij.T)
                    grad_b1k_b5j += q1 * tempk          


                    #-----------
                    grad_W2k_W1j += q1 * np.dot(np.kron(temp6k, a1_ik), np.kron(tempj.T, x_ij.T))                                          
                    grad_W2k_b1j += q1 * np.dot(np.kron(temp6k, a1_ik), tempj.T)     
                    grad_W2k_W2j += q1 * np.dot(np.kron(temp6k, a1_ik), np.kron(temp6j.T, a1_ij.T))                                       
                    grad_W2k_b2j += q1 * np.dot(np.kron(temp6k, a1_ik), temp6j.T)                                                           
                    grad_W2k_W3j += q1 * np.dot(np.kron(temp6k, a1_ik), np.kron(temp5j.T, a2_ij.T))  
                    grad_W2k_b3j += q1 * np.dot(np.kron(temp6k, a1_ik), temp5j.T)                                                 
                    grad_W2k_W4j += q1 * np.dot(np.kron(temp6k, a1_ik), np.kron(temp4j.T, a3_ij.T))  
                    grad_W2k_b4j += q1 * np.dot(np.kron(temp6k, a1_ik), temp4j.T)                                                       
                    grad_W2k_W5j += q1 * np.dot(np.kron(temp6k, a1_ik), a4_ij.T)  
                    grad_W2k_b5j += q1 * np.kron(temp6k, a1_ik)                                                                          

                    #-----------
                    grad_b2k_W1j += q1 * np.dot(temp6k, np.kron(tempj.T, x_ij.T))                                          
                    grad_b2k_b1j += q1 * np.dot(temp6k, tempj.T)     
                    grad_b2k_W2j += q1 * np.dot(temp6k, np.kron(temp6j.T, a1_ij.T))                                       
                    grad_b2k_b2j += q1 * np.dot(temp6k, temp6j.T)                                                           
                    grad_b2k_W3j += q1 * np.dot(temp6k, np.kron(temp5j.T, a2_ij.T))  
                    grad_b2k_b3j += q1 * np.dot(temp6k, temp5j.T)                                                 
                    grad_b2k_W4j += q1 * np.dot(temp6k, np.kron(temp4j.T, a3_ij.T))  
                    grad_b2k_b4j += q1 * np.dot(temp6k, temp4j.T)                                                       
                    grad_b2k_W5j += q1 * np.dot(temp6k, a4_ij.T)  
                    grad_b2k_b5j += q1 * temp6k 

                    #-----------
                    grad_W3k_W1j += q1 * np.dot(np.kron(temp5k, a2_ik), np.kron(tempj.T, x_ij.T))                                          
                    grad_W3k_b1j += q1 * np.dot(np.kron(temp5k, a2_ik), tempj.T)     
                    grad_W3k_W2j += q1 * np.dot(np.kron(temp5k, a2_ik), np.kron(temp6j.T, a1_ij.T))                                       
                    grad_W3k_b2j += q1 * np.dot(np.kron(temp5k, a2_ik), temp6j.T)  
                    grad_W3k_W3j += q1 * np.dot(np.kron(temp5k, a2_ik), np.kron(temp5j.T, a2_ij.T))                                       
                    grad_W3k_b3j += q1 * np.dot(np.kron(temp5k, a2_ik), temp5j.T)                                                       
                    grad_W3k_W4j += q1 * np.dot(np.kron(temp5k, a2_ik), np.kron(temp4j.T, a3_ij.T))   
                    grad_W3k_b4j += q1 * np.dot(np.kron(temp5k, a2_ik), temp4j.T)                                                     
                    grad_W3k_W5j += q1 * np.dot(np.kron(temp5k, a2_ik), a4_ij.T) 
                    grad_W3k_b5j += q1 * np.kron(temp5k, a2_ik) 

                    #-----------
                    grad_b3k_W1j += q1 * np.dot(temp5k, np.kron(tempj.T, x_ij.T))                                          
                    grad_b3k_b1j += q1 * np.dot(temp5k, tempj.T)     
                    grad_b3k_W2j += q1 * np.dot(temp5k, np.kron(temp6j.T, a1_ij.T))                                       
                    grad_b3k_b2j += q1 * np.dot(temp5k, temp6j.T)  
                    grad_b3k_W3j += q1 * np.dot(temp5k, np.kron(temp5j.T, a2_ij.T))                                       
                    grad_b3k_b3j += q1 * np.dot(temp5k, temp5j.T)                                                       
                    grad_b3k_W4j += q1 * np.dot(temp5k, np.kron(temp4j.T, a3_ij.T))   
                    grad_b3k_b4j += q1 * np.dot(temp5k, temp4j.T)                                                     
                    grad_b3k_W5j += q1 * np.dot(temp5k, a4_ij.T) 
                    grad_b3k_b5j += q1 * temp5k 

                    #-------------
                    grad_W4k_W1j += q1 * np.dot(np.kron(temp4k, a3_ik), np.kron(tempj.T, x_ij.T))                                          
                    grad_W4k_b1j += q1 * np.dot(np.kron(temp4k, a3_ik), tempj.T)     
                    grad_W4k_W2j += q1 * np.dot(np.kron(temp4k, a3_ik), np.kron(temp6j.T, a1_ij.T))                                       
                    grad_W4k_b2j += q1 * np.dot(np.kron(temp4k, a3_ik), temp6j.T)  
                    grad_W4k_W3j += q1 * np.dot(np.kron(temp4k, a3_ik), np.kron(temp5j.T, a2_ij.T))                                       
                    grad_W4k_b3j += q1 * np.dot(np.kron(temp4k, a3_ik), temp5j.T)   
                    grad_W4k_W4j += q1 * np.dot(np.kron(temp4k, a3_ik), np.kron(temp4j.T, a3_ij.T))                                    
                    grad_W4k_b4j += q1 * np.dot(np.kron(temp4k, a3_ik), temp4j.T)                                                        
                    grad_W4k_W5j += q1 * np.dot(np.kron(temp4k, a3_ik), a4_ij.T)
                    grad_W4k_b5j += q1 * np.kron(temp4k, a3_ik) 

                    #-------------
                    grad_b4k_W1j += q1 * np.dot(temp4k, np.kron(tempj.T, x_ij.T))                                          
                    grad_b4k_b1j += q1 * np.dot(temp4k, tempj.T)     
                    grad_b4k_W2j += q1 * np.dot(temp4k, np.kron(temp6j.T, a1_ij.T))                                       
                    grad_b4k_b2j += q1 * np.dot(temp4k, temp6j.T)  
                    grad_b4k_W3j += q1 * np.dot(temp4k, np.kron(temp5j.T, a2_ij.T))                                       
                    grad_b4k_b3j += q1 * np.dot(temp4k, temp5j.T)   
                    grad_b4k_W4j += q1 * np.dot(temp4k, np.kron(temp4j.T, a3_ij.T))                                    
                    grad_b4k_b4j += q1 * np.dot(temp4k, temp4j.T)                                                        
                    grad_b4k_W5j += q1 * np.dot(temp4k, a4_ij.T)
                    grad_b4k_b5j += q1 * temp4k 

                    #-------------
                    grad_W5k_W1j += q1 * np.dot(a4_ik, np.kron(tempj.T, x_ij.T))                                          
                    grad_W5k_b1j += q1 * np.dot(a4_ik, tempj.T)     
                    grad_W5k_W2j += q1 * np.dot(a4_ik, np.kron(temp6j.T, a1_ij.T))                                       
                    grad_W5k_b2j += q1 * np.dot(a4_ik, temp6j.T)  
                    grad_W5k_W3j += q1 * np.dot(a4_ik, np.kron(temp5j.T, a2_ij.T))                                       
                    grad_W5k_b3j += q1 * np.dot(a4_ik, temp5j.T)   
                    grad_W5k_W4j += q1 * np.dot(a4_ik, np.kron(temp4j.T, a3_ij.T))                                    
                    grad_W5k_b4j += q1 * np.dot(a4_ik, temp4j.T)                                                        
                    grad_W5k_W5j += q1 * np.dot(a4_ik, a4_ij.T)
                    grad_W5k_b5j += q1 * a4_ik                                                                                 
            

                    #-------------
                    grad_b5k_W1j += q1 * np.kron(tempj.T, x_ij.T)                                          
                    grad_b5k_b1j += q1 * tempj.T     
                    grad_b5k_W2j += q1 * np.kron(temp6j.T, a1_ij.T)                                       
                    grad_b5k_b2j += q1 * temp6j.T  
                    grad_b5k_W3j += q1 * np.kron(temp5j.T, a2_ij.T)                                       
                    grad_b5k_b3j += q1 * temp5j.T   
                    grad_b5k_W4j += q1 * np.kron(temp4j.T, a3_ij.T)                                    
                    grad_b5k_b4j += q1 * temp4j.T                                                        
                    grad_b5k_W5j += q1 * a4_ij.T
                    grad_b5k_b5j += q1                                                                                
                                                                                             

            print(k, j)
            Hessian['scale'+str(k)+str(j)] = np.block([
                [grad_W1k_W1j, grad_W1k_b1j, grad_W1k_W2j, grad_W1k_b2j, grad_W1k_W3j, grad_W1k_b3j, grad_W1k_W4j, grad_W1k_b4j, grad_W1k_W5j, grad_W1k_b5j],
                [grad_b1k_W1j, grad_b1k_b1j, grad_b1k_W2j, grad_b1k_b2j, grad_b1k_W3j, grad_b1k_b3j, grad_b1k_W4j, grad_b1k_b4j, grad_b1k_W5j, grad_b1k_b5j],
                [grad_W2k_W1j, grad_W2k_b1j, grad_W2k_W2j, grad_W2k_b2j, grad_W2k_W3j, grad_W2k_b3j, grad_W2k_W4j, grad_W2k_b4j, grad_W2k_W5j, grad_W2k_b5j],
                [grad_b2k_W1j, grad_b2k_b1j, grad_b2k_W2j, grad_b2k_b2j, grad_b2k_W3j, grad_b2k_b3j, grad_b2k_W4j, grad_b2k_b4j, grad_b2k_W5j, grad_b2k_b5j],
                [grad_W3k_W1j, grad_W3k_b1j, grad_W3k_W2j, grad_W3k_b2j, grad_W3k_W3j, grad_W3k_b3j, grad_W3k_W4j, grad_W3k_b4j, grad_W3k_W5j, grad_W3k_b5j],
                [grad_b3k_W1j, grad_b3k_b1j, grad_b3k_W2j, grad_b3k_b2j, grad_b3k_W3j, grad_b3k_b3j, grad_b3k_W4j, grad_b3k_b4j, grad_b3k_W5j, grad_b3k_b5j],
                [grad_W4k_W1j, grad_W4k_b1j, grad_W4k_W2j, grad_W4k_b2j, grad_W4k_W3j, grad_W4k_b3j, grad_W4k_W4j, grad_W4k_b4j, grad_W4k_W5j, grad_W4k_b5j], 
                [grad_b4k_W1j, grad_b4k_b1j, grad_b4k_W2j, grad_b4k_b2j, grad_b4k_W3j, grad_b4k_b3j, grad_b4k_W4j, grad_b4k_b4j, grad_b4k_W5j, grad_b4k_b5j],
                [grad_W5k_W1j, grad_W5k_b1j, grad_W5k_W2j, grad_W5k_b2j, grad_W5k_W3j, grad_W5k_b3j, grad_W5k_W4j, grad_W5k_b4j, grad_W5k_W5j, grad_W5k_b5j],
                [grad_b5k_W1j, grad_b5k_b1j, grad_b5k_W2j, grad_b5k_b2j, grad_b5k_W3j, grad_b5k_b3j, grad_b5k_W4j, grad_b5k_b4j, grad_b5k_W5j, grad_b5k_b5j]
                ])


    MS_Hessian = np.block([
        [Hessian['scale'+str(1)+str(1)], Hessian['scale'+str(1)+str(2)], Hessian['scale'+str(1)+str(3)], Hessian['scale'+str(1)+str(4)]],
        [Hessian['scale'+str(1)+str(2)].T, Hessian['scale'+str(2)+str(2)], Hessian['scale'+str(2)+str(3)], Hessian['scale'+str(2)+str(4)]],
        [Hessian['scale'+str(1)+str(3)].T, Hessian['scale'+str(2)+str(3)].T, Hessian['scale'+str(3)+str(3)], Hessian['scale'+str(3)+str(4)]],
        [Hessian['scale'+str(1)+str(4)].T, Hessian['scale'+str(2)+str(4)].T, Hessian['scale'+str(3)+str(4)].T, Hessian['scale'+str(4)+str(4)]]
        ])

        
    return MS_Hessian









