import jax.numpy as np


def relu_grad(x):
    return np.where(x >= 0, 1, 0)


def compute_Hessian_one(opt, grade, params, z1, a1, output, u1, u2, train_x, train_y):
    '''
    loss(Theta, u1, u2) = 1/2 sum_{s, t = 1}^n (N(theta; x_{st}) - f_{st})^2 + 
                          lambda*(|u_{1st}| + |u_{2st}|) +
                          beta/2( 
                          (N(theta; x_{st}) - N(theta; x_{s(t-1)}) - u_{1st})^2 +
                          (N(theta; x_{st}) - N(theta; x_{(s-1)t}) - u_{2st})^2 
                          )
    '''

    W1, b1 = params[0]                      
    W2, b2 = params[1]  

    beta = opt.beta
    input_dim = opt.num_channel["grade"+str(grade)][0]
    m1 = opt.num_channel["grade"+str(grade)][1]
    print(f"input_dim: {input_dim}, m1: {m1}")
    
    grad_W1W1 = np.zeros((input_dim * m1, input_dim * m1))
    grad_W1b1 = np.zeros((input_dim * m1, m1))
    grad_W1W2 = np.zeros((input_dim * m1, m1))
    grad_W1b2 = np.zeros((input_dim * m1, 1))

    grad_b1b1 = np.zeros((m1, m1))
    grad_b1W2 = np.zeros((m1, m1))
    grad_b1b2 = np.zeros((m1, 1))


    grad_W2W2 = np.zeros((m1, m1))
    grad_W2b2 = np.zeros((m1, 1))

    grad_b2b2 = np.zeros((1, 1))

    for s in range(opt.nx):
        for t in range(opt.nx):
            i = s*opt.nx + t
            x_i = (train_x[:,i]).reshape(input_dim, 1)
            y_i = train_y[:,i]
            z1_i = (z1[:,i]).reshape(m1, 1)
            a1_i = (a1[:,i]).reshape(m1, 1)
            e = (output[:, i]).reshape(1, 1) - y_i
            grad_diag_z1 = np.diag(relu_grad(z1_i.flatten()))
            temp = (np.dot(grad_diag_z1, W2)).reshape((m1, 1)) 
            temp_xi = np.kron(temp, x_i)
            gradz1_xi = np.kron(grad_diag_z1,  x_i)

           #-----------
            grad_W1W1 += np.dot(temp_xi, temp_xi.T)                                           #(d*m1, d*m1)
            grad_W1b1 += np.dot(temp_xi, temp.T)                                             #(d*m1, m1)
            grad_W1W2 += np.dot(temp_xi, a1_i.T) +  gradz1_xi * e           #(d*m1, m1)
            grad_W1b2 += temp_xi                                                              #(d*m1, 1)
            #-----------                                                                                            
            grad_b1b1 += np.dot(temp, temp.T)                                                #(m1, m1)
            grad_b1W2 += np.dot(temp, a1_i.T) + grad_diag_z1 * e                              #(m1, m1) 
            grad_b1b2 += temp                                                                 #(m1, 1)
            #-----------
            grad_W2W2 += np.dot(a1_i,  a1_i.T)                                                 #(m1, m1)  
            grad_W2b2 += a1_i                                                                  #(m1, 1) 
            #------------
            grad_b2b2 += 1   

            if (s==0 and t!=0) or (s!=0 and t==0):
                if s==0:
                    i0 = s*opt.nx + t-1
                    e0 = output[:, i] - output[:, i0] - u1[:, i]
                else:
                    i0 = (s-1)*opt.nx + t
                    e0 = output[:, i] - output[:, i0] - u2[:, i]
                
                x_i0 = (train_x[:,i0]).reshape(input_dim, 1)
                z1_i0 = (z1[:,i0]).reshape(m1, 1)
                a1_i0 = (a1[:,i0]).reshape(m1, 1)
                
                grad_diag_z10 = np.diag(relu_grad(z1_i0.flatten()))
                temp00 = (np.dot(grad_diag_z10, W2)).reshape((m1, 1))  

                diff0_temp_xi = temp_xi - np.kron(temp00, x_i0)
                diff0_temp = temp - temp00
                diff0_a1 = a1_i - a1_i0
                
                diff0_gradz1_xi = gradz1_xi - np.kron(grad_diag_z10,  x_i0)
                diff0_gradz1 = grad_diag_z1 - grad_diag_z10

                #-----------
                grad_W1W1 += beta*np.dot(diff0_temp_xi, diff0_temp_xi.T)       
                grad_W1b1 += beta*np.dot(diff0_temp_xi, diff0_temp.T)      
                grad_W1W2 += beta*(diff0_gradz1_xi*e0 + np.dot(diff0_temp_xi, diff0_a1.T))                                                                                                          
                #-----------                                                                                            
                grad_b1b1 += beta*np.dot(diff0_temp, diff0_temp.T)                                          
                grad_b1W2 += beta*(diff0_gradz1*e0 + np.dot(diff0_temp, diff0_a1.T))                                                          
        
                #-----------
                grad_W2W2 += beta*np.dot(diff0_a1, diff0_a1.T)                                                                                                       
            
            if s!=0 and t!=0:
                #------------
                i0 = s*opt.nx + t-1
                i1 = (s-1)*opt.nx + t

                #------------
                x_i0 = (train_x[:,i0]).reshape(input_dim, 1)
                z1_i0 = (z1[:,i0]).reshape(m1, 1)
                a1_i0 = (a1[:,i0]).reshape(m1, 1)
                grad_diag_z10 = np.diag(relu_grad(z1_i0.flatten()))
                temp00 = (np.dot(grad_diag_z10, W2)).reshape((m1, 1))  
                e0 = output[:, i] - output[:, i0] - u1[:, i]

                #------------
                x_i1 = (train_x[:,i1]).reshape(input_dim, 1)
                z1_i1 = (z1[:,i1]).reshape(m1, 1)
                a1_i1 = (a1[:,i1]).reshape(m1, 1)
                grad_diag_z11 = np.diag(relu_grad(z1_i1.flatten()))
                temp01 = (np.dot(grad_diag_z11, W2)).reshape((m1, 1)) 
                e1 = output[:, i] - output[:, i1] - u2[:, i]
 

                #-------------
                diff0_temp_xi = temp_xi - np.kron(temp00, x_i0)
                diff0_temp = temp - temp00
                diff0_a1 = a1_i - a1_i0
                
                diff0_gradz1_xi = gradz1_xi - np.kron(grad_diag_z10,  x_i0)
                diff0_gradz1 = grad_diag_z1 - grad_diag_z10
                
                #--------------
                diff1_temp_xi = temp_xi - np.kron(temp01, x_i1)
                diff1_temp = temp - temp01
                diff1_a1 = a1_i - a1_i1
                
                diff1_gradz1_xi = gradz1_xi - np.kron(grad_diag_z11,  x_i1)
                diff1_gradz1 = grad_diag_z1 - grad_diag_z11


                #-----------
                grad_W1W1 += beta*(np.dot(diff0_temp_xi, diff0_temp_xi.T)+np.dot(diff1_temp_xi, diff1_temp_xi.T))       
                grad_W1b1 += beta*(np.dot(diff0_temp_xi, diff0_temp.T)+np.dot(diff1_temp_xi, diff1_temp.T))              
                grad_W1W2 += beta*(diff0_gradz1_xi*e0 + np.dot(diff0_temp_xi, diff0_a1.T)+
                                   diff1_gradz1_xi*e1 + np.dot(diff1_temp_xi, diff1_a1.T))                                                                                                                                                                 
                #-----------                                                                                            
                grad_b1b1 += beta*(np.dot(diff0_temp, diff0_temp.T) + np.dot(diff1_temp, diff1_temp.T))                                
                grad_b1W2 += beta*(diff0_gradz1*e0 + np.dot(diff0_temp, diff0_a1.T)+
                                   diff1_gradz1*e1 + np.dot(diff1_temp, diff1_a1.T))
                
                #-----------
                grad_W2W2 += beta*(np.dot(diff0_a1, diff0_a1.T) + np.dot(diff1_a1, diff1_a1.T))                                         
                                                               

        
        
    grad_b1W1 = grad_W1b1.T                                                                                   
    
    grad_W2W1 = grad_W1W2.T                                                                                   
    grad_W2b1 = grad_b1W2.T                                                                                    

    grad_b2W1 = grad_W1b2.T                                                                                    
    grad_b2b1 = grad_b1b2.T                                                                                    
    grad_b2W2 = grad_W2b2.T                                                                                    

    
    
    Hessian = np.block([[grad_W1W1, grad_W1b1, grad_W1W2, grad_W1b2],
                        [grad_b1W1, grad_b1b1, grad_b1W2, grad_b1b2],
                        [grad_W2W1, grad_W2b1, grad_W2W2, grad_W2b2],
                        [grad_b2W1, grad_b2b1, grad_b2W2, grad_b2b2]])
    
    return Hessian


def compute_Hessian_four(opt, params, z1, z2, z3, z4, a1, a2, a3, a4, u1, u2, output, train_x, train_y):


    W1, b1 = params[0]                      
    W2, b2 = params[1]  
    W3, b3 = params[2]
    W4, b4 = params[3]
    W5, b5 = params[4]

    beta = opt.beta

    
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
    
    grad_b5b5 = np.zeros((1, 1))


    for s in range(opt.nx):
        for t in range(opt.nx):
            i = s*opt.nx + t
            x_i = (train_x[:,i]).reshape(input_dim, 1)
            y_i = train_y[:,i]
            z1_i = (z1[:,i]).reshape(m1, 1)
            z2_i = (z2[:,i]).reshape(m2, 1)
            z3_i = (z3[:,i]).reshape(m3, 1)
            z4_i = (z4[:,i]).reshape(m4, 1)
            a1_i = (a1[:,i]).reshape(m1, 1)
            a2_i = (a2[:,i]).reshape(m2, 1)
            a3_i = (a3[:,i]).reshape(m3, 1)
            a4_i = (a4[:,i]).reshape(m4, 1)
            e = (output[:, i]).reshape(1, 1) - y_i
            
            #--------
            grad_diag_z1 = np.diag(relu_grad(z1_i.flatten()))
            grad_diag_z2 = np.diag(relu_grad(z2_i.flatten()))
            grad_diag_z3 = np.diag(relu_grad(z3_i.flatten()))
            grad_diag_z4 = np.diag(relu_grad(z4_i.flatten()))
            temp1 = (np.dot( grad_diag_z1, W2)).reshape((m1, m2))                                        #(m1, m2)  Z_1
            temp2 = (np.dot( grad_diag_z2, W3)).reshape((m2, m3))                                        #(m2, m3)  Z_2
            temp3 = (np.dot( grad_diag_z3, W4)).reshape((m3, m4))                                        #(m3, m4)  Z_3
            temp4 = (np.dot( grad_diag_z4, W5)).reshape((m4, 1))                                         #(m4, 1)   Z_4
            temp5 = np.dot(temp3, temp4)                                                                 #(m3, 1) Z_3*Z_4
            temp6 = np.dot(temp2, temp5)                                                                 #(m2, 1) Z_2*Z_3*Z4
            temp = np.dot(temp1, temp6)                                                                  #(m1, 1) Z_1*Z_3*Z3*Z4
            temp7 = np.dot(temp1, temp2)                                                                 #(m1, m2) Z_1*Z_2
            temp8 = np.dot(temp7, temp3)                                                                 #(m1, m3) Z_1*Z_2*Z_3
            temp9 = np.dot(temp2, temp3)                                                                 #(m2, m3) Z_2*Z_3

            #--------
            temp_xi = np.kron(temp, x_i)
            temp6_a1 = np.kron(temp6, a1_i)
            temp6_gradz1 = np.kron(temp6.T, grad_diag_z1)
            temp5_a2 = np.kron(temp5, a2_i)
            temp5_temp1_gradz2 = np.kron(temp5.T, np.dot(temp1, grad_diag_z2))
            temp4_a3 = np.kron(temp4, a3_i)
            temp4_temp7_gradz3 = np.kron(temp4.T, np.dot(temp7, grad_diag_z3))
            temp8_gradz4 = np.dot(temp8, grad_diag_z4)
            temp5_gradz2 = np.kron(temp5.T, grad_diag_z2)
            temp4_temp3_gradz3 = np.kron(temp4.T, np.dot(temp3, grad_diag_z3))
            temp9_gradz4 = np.dot(temp9, grad_diag_z4)
            temp4_gradz3 = np.kron(temp4.T, grad_diag_z3)
            temp3_gradz4 = np.dot(temp3, grad_diag_z4)


            grad_W1W1 += np.dot(temp_xi, temp_xi.T)                                        
            grad_W1b1 += np.dot(temp_xi, temp.T)                                                       
            grad_W1W2 += np.dot(temp_xi, temp6_a1.T) +  np.kron(temp6_gradz1,  x_i) * e                                   
            grad_W1b2 += np.dot(temp_xi, temp6.T)                                                        
            grad_W1W3 += np.dot(temp_xi, temp5_a2.T) + np.kron(temp5_temp1_gradz2,  x_i) * e
            grad_W1b3 += np.dot(temp_xi, temp5.T)                                                        
            grad_W1W4 += np.dot(temp_xi, temp4_a3.T)  + np.kron(temp4_temp7_gradz3, x_i) * e                        
            grad_W1b4 += np.dot(temp_xi, temp4.T)                                                         
            grad_W1W5 += np.dot(temp_xi, a4_i.T)  + np.kron(temp8_gradz4, x_i) * e                                        
            grad_W1b5 += temp_xi  
            #--------

            #--------                                                                                      
            grad_b1b1 += np.dot(temp, temp.T)                                                                      
            grad_b1W2 += np.dot(temp, temp6_a1.T) + temp6_gradz1 * e                                                       
            grad_b1b2 += np.dot(temp, temp6.T)                                                                        
            grad_b1W3 += np.dot(temp, temp5_a2.T) + temp5_temp1_gradz2 * e                                  
            grad_b1b3 += np.dot(temp, temp5.T)                                                                      
            grad_b1W4 += np.dot(temp, temp4_a3.T) + temp4_temp7_gradz3 * e                                    
            grad_b1b4 += np.dot(temp, temp4.T)                                                                     
            grad_b1W5 += np.dot(temp, a4_i.T) + temp8_gradz4 * e                                                        
            grad_b1b5 += temp  
            #--------

            #--------
            grad_W2W2 += np.dot(temp6_a1, temp6_a1.T)                                    
            grad_W2b2 += np.dot(temp6_a1, temp6.T)                                                          
            grad_W2W3 += np.dot(temp6_a1, temp5_a2.T) + np.kron(temp5_gradz2, a1_i) * e                                          
            grad_W2b3 += np.dot(temp6_a1, temp5.T)                                                           
            grad_W2W4 += np.dot(temp6_a1, temp4_a3.T) + np.kron(temp4_temp3_gradz3, a1_i) * e                      
            grad_W2b4 += np.dot(temp6_a1, temp4.T)                                                            
            grad_W2W5 += np.dot(temp6_a1, a4_i.T) + np.kron(temp9_gradz4, a1_i) * e                                   
            grad_W2b5 += temp6_a1
            #--------

            #--------
            grad_b2b2 += np.dot(temp6, temp6.T)                                                                      
            grad_b2W3 += np.dot(temp6,  temp5_a2.T) + temp5_gradz2 * e                                                       
            grad_b2b3 += np.dot(temp6, temp5.T)                                                                       
            grad_b2W4 += np.dot(temp6, temp4_a3.T) + temp4_temp3_gradz3 * e                                            
            grad_b2b4 += np.dot(temp6, temp4.T)                                                                        
            grad_b2W5 += np.dot(temp6,  a4_i.T) + temp9_gradz4 * e                                                         
            grad_b2b5 += temp6     
            #--------

            #--------
            grad_W3W3 += np.dot(temp5_a2, temp5_a2.T)                                       
            grad_W3b3 += np.dot(temp5_a2, temp5.T)                                                         
            grad_W3W4 += np.dot(temp5_a2, temp4_a3.T) + np.kron(temp4_gradz3, a2_i) * e                                       
            grad_W3b4 += np.dot(temp5_a2, temp4.T)                                                           
            grad_W3W5 += np.dot(temp5_a2, a4_i.T) + np.kron(temp3_gradz4, a2_i) * e                                              
            grad_W3b5 += temp5_a2  
            #--------

            #--------
            grad_b3b3 += np.dot(temp5, temp5.T)                                                                        
            grad_b3W4 += np.dot(temp5, temp4_a3.T) + temp4_gradz3 * e                                                        
            grad_b3b4 += np.dot(temp5, temp4.T)                                                                     
            grad_b3W5 += np.dot(temp5, a4_i.T) + temp3_gradz4 * e
            grad_b3b5 += temp5
            #--------
            
            #--------
            grad_W4W4 += np.dot(temp4_a3, temp4_a3.T)                                        
            grad_W4b4 += np.dot(temp4_a3, temp4.T)
            grad_W4W5 += np.dot(temp4_a3, a4_i.T) + np.kron(grad_diag_z4, a3_i) * e
            grad_W4b5 += temp4_a3 
            #--------

            #--------
            grad_b4b4 += np.dot(temp4, temp4.T)                                                                       
            grad_b4W5 += np.dot(temp4, a4_i.T) + grad_diag_z4 * e 
            grad_b4b5 += temp4
            #--------

            #--------
            grad_W5W5 += np.dot(a4_i, a4_i.T)                                                                          
            grad_W5b5 += a4_i  
            #--------

            #--------
            grad_b5b5 += 1
            #--------


            if (s==0 and t!=0) or (s!=0 and t==0):
                if s==0:
                    i0 = s*opt.nx + t-1
                    e0 = output[:, i] - output[:, i0] - u1[:, i]
                else:
                    i0 = (s-1)*opt.nx + t
                    e0 = output[:, i] - output[:, i0] - u2[:, i]

                x_i0 = (train_x[:,i0]).reshape(input_dim, 1)
                z1_i0 = (z1[:,i0]).reshape(m1, 1)
                z2_i0 = (z2[:,i0]).reshape(m2, 1)
                z3_i0 = (z3[:,i0]).reshape(m3, 1)
                z4_i0 = (z4[:,i0]).reshape(m4, 1)
                a1_i0 = (a1[:,i0]).reshape(m1, 1)
                a2_i0 = (a2[:,i0]).reshape(m2, 1)
                a3_i0 = (a3[:,i0]).reshape(m3, 1)
                a4_i0 = (a4[:,i0]).reshape(m4, 1)
                
                #------------
                grad_diag_z10 = np.diag(relu_grad(z1_i0.flatten()))
                grad_diag_z20 = np.diag(relu_grad(z2_i0.flatten()))
                grad_diag_z30 = np.diag(relu_grad(z3_i0.flatten()))
                grad_diag_z40 = np.diag(relu_grad(z4_i0.flatten()))
                temp10 = (np.dot(grad_diag_z10, W2)).reshape((m1, m2))                                         #(m1, m2)  Z_1
                temp20 = (np.dot(grad_diag_z20, W3)).reshape((m2, m3))                                         #(m2, m3)  Z_2
                temp30 = (np.dot(grad_diag_z30, W4)).reshape((m3, m4))                                         #(m3, m4)  Z_3
                temp40 = (np.dot(grad_diag_z40, W5)).reshape((m4, 1))                                          #(m4, 1)   Z_4
                temp50 = np.dot(temp30, temp40)                                                                #(m3, 1) Z_3*Z_4
                temp60 = np.dot(temp20, temp50)                                                                #(m2, 1) Z_2*Z_3*Z4
                temp00 = np.dot(temp10, temp60)                                                                #(m1, 1) Z_1*Z_3*Z3*Z4
                temp70 = np.dot(temp10, temp20)                                                                #(m1, m2) Z_1*Z_2
                temp80 = np.dot(temp70, temp30)                                                                #(m1, m3) Z_1*Z_2*Z_3
                temp90 = np.dot(temp20, temp30)                                                                #(m2, m3) Z_2*Z_3
                #------------

                #------------
                diff0_temp_xi = temp_xi - np.kron(temp00, x_i0)
                diff0_temp = temp - temp00
                diff0_temp6_gradz1_xi = np.kron(temp6_gradz1,  x_i) - np.kron(np.kron(temp60.T, grad_diag_z10), x_i0)
                diff0_temp6_a1 = np.kron(temp6, a1_i) - np.kron(temp60, a1_i0)
                diff0_temp6 = temp6 - temp60
                diff0_temp5_temp1_gradz2_xi = np.kron(temp5_temp1_gradz2, x_i) - np.kron(np.kron(temp50.T, np.dot(temp10, grad_diag_z20)),x_i0)
                diff0_temp5_a2 = np.kron(temp5, a2_i) - np.kron(temp50, a2_i0)
                diff0_temp5 = temp5 - temp50        
                diff0_temp4_temp7_gradz3_xi = np.kron(temp4_temp7_gradz3, x_i) - np.kron(np.kron(temp40.T, np.dot(temp70, grad_diag_z30)), x_i0)
                diff0_temp4_a3 = np.kron(temp4, a3_i) - np.kron(temp40, a3_i0)
                diff0_temp4 = temp4 - temp40  
                diff0_temp8_gradz4_xi = np.kron(temp8_gradz4, x_i) - np.kron(np.dot(temp80, grad_diag_z40), x_i0)
                diff0_a4 = a4_i - a4_i0                                                                                          
                                                                                                        
                grad_W1W1 += beta * np.dot(diff0_temp_xi, diff0_temp_xi.T)                                       
                grad_W1b1 += beta * np.dot(diff0_temp_xi, diff0_temp.T)                                                       
                grad_W1W2 += beta * (diff0_temp6_gradz1_xi * e0 + np.dot(diff0_temp_xi, diff0_temp6_a1.T))                                  
                grad_W1b2 += beta * np.dot(diff0_temp_xi, diff0_temp6.T)                                                  
                grad_W1W3 += beta * (diff0_temp5_temp1_gradz2_xi * e0 + np.dot(diff0_temp_xi, diff0_temp5_a2.T))
                grad_W1b3 += beta * np.dot(diff0_temp_xi, diff0_temp5.T)                                                      
                grad_W1W4 += beta * (diff0_temp4_temp7_gradz3_xi * e0 + np.dot(diff0_temp_xi, diff0_temp4_a3.T))                       
                grad_W1b4 += beta * np.dot(diff0_temp_xi, diff0_temp4.T)                                                         
                grad_W1W5 += beta * (diff0_temp8_gradz4_xi * e0 + np.dot(diff0_temp_xi, diff0_a4.T))
                #------------

                #------------
                diff0_temp6_gradz1 = temp6_gradz1 - np.kron(temp60.T, grad_diag_z10)
                diff0_temp5_temp1_gradz2 = temp5_temp1_gradz2 - np.kron(temp50.T, np.dot(temp10, grad_diag_z20))
                diff0_temp4_temp7_gradz3 = temp4_temp7_gradz3 - np.kron(temp40.T, np.dot(temp70, grad_diag_z30))
                diff0_temp8_gradz4 = temp8_gradz4 - np.dot(temp80, grad_diag_z40)

                grad_b1b1 += beta * np.dot(diff0_temp, diff0_temp.T)                                                       
                grad_b1W2 += beta * (diff0_temp6_gradz1 * e0 + np.dot(diff0_temp, diff0_temp6_a1.T))                                  
                grad_b1b2 += beta * np.dot(diff0_temp, diff0_temp6.T)                                                  
                grad_b1W3 += beta * (diff0_temp5_temp1_gradz2 * e0 + np.dot(diff0_temp, diff0_temp5_a2.T))
                grad_b1b3 += beta * np.dot(diff0_temp, diff0_temp5.T)                                                      
                grad_b1W4 += beta * (diff0_temp4_temp7_gradz3 * e0 + np.dot(diff0_temp, diff0_temp4_a3.T))                       
                grad_b1b4 += beta * np.dot(diff0_temp, diff0_temp4.T)                                                         
                grad_b1W5 += beta * (diff0_temp8_gradz4 * e0 + np.dot(diff0_temp, diff0_a4.T))   
                #------------

                #------------
                diff0_temp5_gradz2_a1 = np.kron(temp5_gradz2, a1_i) - np.kron(np.kron(temp50.T, grad_diag_z20), a1_i0)
                diff0_temp4_temp3_gradz3_a1 = np.kron(temp4_temp3_gradz3, a1_i) - np.kron(np.kron(temp40.T, np.dot(temp30, grad_diag_z30)), a1_i0)
                diff0_temp9_gradz4_a1 = np.kron(temp9_gradz4, a1_i) - np.kron(np.dot(temp90, grad_diag_z40), a1_i0)
                
                grad_W2W2 += beta * np.dot(diff0_temp6_a1, diff0_temp6_a1.T)                               
                grad_W2b2 += beta * np.dot(diff0_temp6_a1, diff0_temp6.T)                                                  
                grad_W2W3 += beta * (diff0_temp5_gradz2_a1 * e0 + np.dot(diff0_temp6_a1, diff0_temp5_a2.T))
                grad_W2b3 += beta * np.dot(diff0_temp6_a1, diff0_temp5.T)                                                      
                grad_W2W4 += beta * (diff0_temp4_temp3_gradz3_a1 * e0 + np.dot(diff0_temp6_a1, diff0_temp4_a3.T))                       
                grad_W2b4 += beta * np.dot(diff0_temp6_a1, diff0_temp4.T)                                                         
                grad_W2W5 += beta * (diff0_temp9_gradz4_a1 * e0 + np.dot(diff0_temp6_a1, diff0_a4.T))
                #------------

                #------------
                diff0_temp5_gradz2 = temp5_gradz2 - np.kron(temp50.T, grad_diag_z20)
                diff0_temp4_temp3_gradz3 = temp4_temp3_gradz3 - np.kron(temp40.T, np.dot(temp30, grad_diag_z30))
                diff0_temp9_gradz4 = temp9_gradz4 - np.dot(temp90, grad_diag_z40)               

                grad_b2b2 += beta * np.dot(diff0_temp6, diff0_temp6.T)     
                grad_b2W3 += beta * (diff0_temp5_gradz2 * e0 + np.dot(diff0_temp6, diff0_temp5_a2.T))
                grad_b2b3 += beta * np.dot(diff0_temp6, diff0_temp5.T)                                                      
                grad_b2W4 += beta * (diff0_temp4_temp3_gradz3 * e0 + np.dot(diff0_temp6, diff0_temp4_a3.T))                       
                grad_b2b4 += beta * np.dot(diff0_temp6, diff0_temp4.T)                                                         
                grad_b2W5 += beta * (diff0_temp9_gradz4 * e0 + np.dot(diff0_temp6, diff0_a4.T))
                #------------

                #------------
                diff0_temp4_gradz3_a2 = np.kron(temp4_gradz3, a2_i) - np.kron(np.kron(temp40.T, grad_diag_z30), a2_i0)
                diff0_temp3_gradz4_a2 = np.kron(temp3_gradz4, a2_i) - np.kron(np.dot(temp30, grad_diag_z40), a2_i0)

                grad_W3W3 += beta * np.dot(diff0_temp5_a2, diff0_temp5_a2.T)
                grad_W3b3 += beta * np.dot(diff0_temp5_a2, diff0_temp5.T)                                                      
                grad_W3W4 += beta * (diff0_temp4_gradz3_a2 * e0 + np.dot(diff0_temp5_a2, diff0_temp4_a3.T))                       
                grad_W3b4 += beta * np.dot(diff0_temp5_a2, diff0_temp4.T)                                                         
                grad_W3W5 += beta * (diff0_temp3_gradz4_a2 * e0 + np.dot(diff0_temp5_a2, diff0_a4.T))
                #------------

                #------------
                diff0_temp4_gradz3 = temp4_gradz3 - np.kron(temp40.T, grad_diag_z30)
                diff0_temp3_gradz4 = temp3_gradz4 - np.dot(temp30, grad_diag_z40)

                grad_b3b3 += beta * np.dot(diff0_temp5, diff0_temp5.T)                                                      
                grad_b3W4 += beta * (diff0_temp4_gradz3 * e0 + np.dot(diff0_temp5, diff0_temp4_a3.T))                       
                grad_b3b4 += beta * np.dot(diff0_temp5, diff0_temp4.T)                                                         
                grad_b3W5 += beta * (diff0_temp3_gradz4 * e0 + np.dot(diff0_temp5, diff0_a4.T))
                #------------

                #------------
                diff0_gradz4_a3 = np.kron(grad_diag_z4, a3_i) - np.kron(grad_diag_z40, a3_i0)
                                                
                grad_W4W4 += beta * np.dot(diff0_temp4_a3, diff0_temp4_a3.T)                      
                grad_W4b4 += beta * np.dot(diff0_temp4_a3, diff0_temp4.T)                                                         
                grad_W4W5 += beta * (diff0_gradz4_a3 * e0 + np.dot(diff0_temp4_a3, diff0_a4.T))
                #------------

                #------------
                diff0_gradz4 = grad_diag_z4 - grad_diag_z40
                     
                grad_b4b4 += beta * np.dot(diff0_temp4, diff0_temp4.T)                                                         
                grad_b4W5 += beta * (diff0_gradz4 * e0 + np.dot(diff0_temp4, diff0_a4.T))
                #------------

                #------------
                grad_W5W5 += beta * np.dot(diff0_a4, diff0_a4.T)
                #------------

            if s!=0 and t!=0:
                i0 = s*opt.nx + t-1
                e0 = output[:, i] - output[:, i0] - u1[:, i]
                i1 = (s-1)*opt.nx + t
                e1 = output[:, i] - output[:, i1] - u2[:, i]

                x_i0 = (train_x[:,i0]).reshape(input_dim, 1)
                z1_i0 = (z1[:,i0]).reshape(m1, 1)
                z2_i0 = (z2[:,i0]).reshape(m2, 1)
                z3_i0 = (z3[:,i0]).reshape(m3, 1)
                z4_i0 = (z4[:,i0]).reshape(m4, 1)
                a1_i0 = (a1[:,i0]).reshape(m1, 1)
                a2_i0 = (a2[:,i0]).reshape(m2, 1)
                a3_i0 = (a3[:,i0]).reshape(m3, 1)
                a4_i0 = (a4[:,i0]).reshape(m4, 1)

                x_i1 = (train_x[:,i1]).reshape(input_dim, 1)
                z1_i1 = (z1[:,i1]).reshape(m1, 1)
                z2_i1 = (z2[:,i1]).reshape(m2, 1)
                z3_i1 = (z3[:,i1]).reshape(m3, 1)
                z4_i1 = (z4[:,i1]).reshape(m4, 1)
                a1_i1 = (a1[:,i1]).reshape(m1, 1)
                a2_i1 = (a2[:,i1]).reshape(m2, 1)
                a3_i1 = (a3[:,i1]).reshape(m3, 1)
                a4_i1 = (a4[:,i1]).reshape(m4, 1)

                #------------
                grad_diag_z10 = np.diag(relu_grad(z1_i0.flatten()))
                grad_diag_z20 = np.diag(relu_grad(z2_i0.flatten()))
                grad_diag_z30 = np.diag(relu_grad(z3_i0.flatten()))
                grad_diag_z40 = np.diag(relu_grad(z4_i0.flatten()))
                temp10 = (np.dot(grad_diag_z10, W2)).reshape((m1, m2))                                         #(m1, m2)  Z_1
                temp20 = (np.dot(grad_diag_z20, W3)).reshape((m2, m3))                                         #(m2, m3)  Z_2
                temp30 = (np.dot(grad_diag_z30, W4)).reshape((m3, m4))                                         #(m3, m4)  Z_3
                temp40 = (np.dot(grad_diag_z40, W5)).reshape((m4, 1))                                          #(m4, 1)   Z_4
                temp50 = np.dot(temp30, temp40)                                                                 #(m3, 1) Z_3*Z_4
                temp60 = np.dot(temp20, temp50)                                                                 #(m2, 1) Z_2*Z_3*Z4
                temp00 = np.dot(temp10, temp60)                                                                  #(m1, 1) Z_1*Z_3*Z3*Z4
                temp70 = np.dot(temp10, temp20)                                                                 #(m1, m2) Z_1*Z_2
                temp80 = np.dot(temp70, temp30)                                                                 #(m1, m3) Z_1*Z_2*Z_3
                temp90 = np.dot(temp20, temp30)                                                                 #(m2, m3) Z_2*Z_3
                #------------

                #------------
                grad_diag_z11 = np.diag(relu_grad(z1_i1.flatten()))
                grad_diag_z21 = np.diag(relu_grad(z2_i1.flatten()))
                grad_diag_z31 = np.diag(relu_grad(z3_i1.flatten()))
                grad_diag_z41 = np.diag(relu_grad(z4_i1.flatten()))
                temp11 = (np.dot(grad_diag_z11, W2)).reshape((m1, m2))                                         #(m1, m2)  Z_1
                temp21 = (np.dot(grad_diag_z21, W3)).reshape((m2, m3))                                         #(m2, m3)  Z_2
                temp31 = (np.dot(grad_diag_z31, W4)).reshape((m3, m4))                                         #(m3, m4)  Z_3
                temp41 = (np.dot(grad_diag_z41, W5)).reshape((m4, 1))                                          #(m4, 1)   Z_4
                temp51 = np.dot(temp31, temp41)                                                                 #(m3, 1) Z_3*Z_4
                temp61 = np.dot(temp21, temp51)                                                                 #(m2, 1) Z_2*Z_3*Z4
                temp01 = np.dot(temp11, temp61)                                                                  #(m1, 1) Z_1*Z_3*Z3*Z4
                temp71 = np.dot(temp11, temp21)                                                                 #(m1, m2) Z_1*Z_2
                temp81 = np.dot(temp71, temp31)                                                                 #(m1, m3) Z_1*Z_2*Z_3
                temp91 = np.dot(temp21, temp31)                                                                 #(m2, m3) Z_2*Z_3
                #------------

                #------------
                diff0_temp_xi = temp_xi - np.kron(temp00, x_i0)
                diff0_temp = temp - temp00
                diff0_temp6_gradz1_xi = np.kron(temp6_gradz1,  x_i) - np.kron(np.kron(temp60.T, grad_diag_z10), x_i0)
                diff0_temp6_a1 = np.kron(temp6, a1_i) - np.kron(temp60, a1_i0)
                diff0_temp6 = temp6 - temp60
                diff0_temp5_temp1_gradz2_xi = np.kron(temp5_temp1_gradz2, x_i) - np.kron(np.kron(temp50.T, np.dot(temp10, grad_diag_z20)), x_i0)
                diff0_temp5_a2 = np.kron(temp5, a2_i) - np.kron(temp50, a2_i0)
                diff0_temp5 = temp5 - temp50        
                diff0_temp4_temp7_gradz3_xi = np.kron(temp4_temp7_gradz3, x_i) - np.kron(np.kron(temp40.T, np.dot(temp70, grad_diag_z30)), x_i0)
                diff0_temp4_a3 = np.kron(temp4, a3_i) - np.kron(temp40, a3_i0)
                diff0_temp4 = temp4 - temp40  
                diff0_temp8_gradz4_xi = np.kron(temp8_gradz4, x_i) - np.kron(np.dot(temp80, grad_diag_z40), x_i0)
                diff0_a4 = a4_i - a4_i0    

                diff1_temp_xi = temp_xi - np.kron(temp01, x_i1)
                diff1_temp = temp - temp01
                diff1_temp6_gradz1_xi = np.kron(temp6_gradz1,  x_i) - np.kron(np.kron(temp61.T, grad_diag_z11), x_i1)
                diff1_temp6_a1 = np.kron(temp6, a1_i) - np.kron(temp61, a1_i1)
                diff1_temp6 = temp6 - temp61
                diff1_temp5_temp1_gradz2_xi = np.kron(temp5_temp1_gradz2, x_i) - np.kron(np.kron(temp51.T, np.dot(temp11, grad_diag_z21)),x_i1)
                diff1_temp5_a2 = np.kron(temp5, a2_i) - np.kron(temp51, a2_i1)
                diff1_temp5 = temp5 - temp51        
                diff1_temp4_temp7_gradz3_xi = np.kron(temp4_temp7_gradz3, x_i) - np.kron(np.kron(temp41.T, np.dot(temp71, grad_diag_z31)), x_i1)
                diff1_temp4_a3 = np.kron(temp4, a3_i) - np.kron(temp41, a3_i1)
                diff1_temp4 = temp4 - temp41  
                diff1_temp8_gradz4_xi = np.kron(temp8_gradz4, x_i) - np.kron(np.dot(temp81, grad_diag_z41), x_i1)
                diff1_a4 = a4_i - a4_i1  
                                                                                                        
                grad_W1W1 += beta * (np.dot(diff0_temp_xi, diff0_temp_xi.T) + np.dot(diff1_temp_xi, diff1_temp_xi.T))                                       
                grad_W1b1 += beta * (np.dot(diff0_temp_xi, diff0_temp.T) + np.dot(diff1_temp_xi, diff1_temp.T))                                                 
                grad_W1W2 += beta * (diff0_temp6_gradz1_xi * e0 + np.dot(diff0_temp_xi, diff0_temp6_a1.T) + 
                                     diff1_temp6_gradz1_xi * e1 + np.dot(diff1_temp_xi, diff1_temp6_a1.T))                                  
                grad_W1b2 += beta * (np.dot(diff0_temp_xi, diff0_temp6.T) + np.dot(diff1_temp_xi, diff1_temp6.T))                                                 
                grad_W1W3 += beta * (diff0_temp5_temp1_gradz2_xi * e0 + np.dot(diff0_temp_xi, diff0_temp5_a2.T)+
                                     diff1_temp5_temp1_gradz2_xi * e1 + np.dot(diff1_temp_xi, diff1_temp5_a2.T))
                grad_W1b3 += beta * (np.dot(diff0_temp_xi, diff0_temp5.T) + np.dot(diff1_temp_xi, diff1_temp5.T))                                                 
                grad_W1W4 += beta * (diff0_temp4_temp7_gradz3_xi * e0 + np.dot(diff0_temp_xi, diff0_temp4_a3.T) + 
                                     diff1_temp4_temp7_gradz3_xi * e1 + np.dot(diff1_temp_xi, diff1_temp4_a3.T))                       
                grad_W1b4 += beta * (np.dot(diff0_temp_xi, diff0_temp4.T) + np.dot(diff1_temp_xi, diff1_temp4.T))                                                 
                grad_W1W5 += beta * (diff0_temp8_gradz4_xi * e0 + np.dot(diff0_temp_xi, diff0_a4.T) + 
                                     diff1_temp8_gradz4_xi * e1 + np.dot(diff1_temp_xi, diff1_a4.T))
                #------------

                #------------
                diff0_temp6_gradz1 = temp6_gradz1 - np.kron(temp60.T, grad_diag_z10)
                diff0_temp5_temp1_gradz2 = temp5_temp1_gradz2 - np.kron(temp50.T, np.dot(temp10, grad_diag_z20))
                diff0_temp4_temp7_gradz3 = temp4_temp7_gradz3 - np.kron(temp40.T, np.dot(temp70, grad_diag_z30))
                diff0_temp8_gradz4 = temp8_gradz4 - np.dot(temp80, grad_diag_z40)

                diff1_temp6_gradz1 = temp6_gradz1 - np.kron(temp61.T, grad_diag_z11)
                diff1_temp5_temp1_gradz2 = temp5_temp1_gradz2 - np.kron(temp51.T, np.dot(temp11, grad_diag_z21))
                diff1_temp4_temp7_gradz3 = temp4_temp7_gradz3 - np.kron(temp41.T, np.dot(temp71, grad_diag_z31))
                diff1_temp8_gradz4 = temp8_gradz4 - np.dot(temp81, grad_diag_z41)

                grad_b1b1 += beta * (np.dot(diff0_temp, diff0_temp.T) + np.dot(diff1_temp, diff1_temp.T))                                                       
                grad_b1W2 += beta * (diff0_temp6_gradz1 * e0 + np.dot(diff0_temp, diff0_temp6_a1.T) + 
                                     diff1_temp6_gradz1 * e1 + np.dot(diff1_temp, diff1_temp6_a1.T))                                  
                grad_b1b2 += beta * (np.dot(diff0_temp, diff0_temp6.T) + np.dot(diff1_temp, diff1_temp6.T))                                                  
                grad_b1W3 += beta * (diff0_temp5_temp1_gradz2 * e0 + np.dot(diff0_temp, diff0_temp5_a2.T) + 
                                     diff1_temp5_temp1_gradz2 * e1 + np.dot(diff1_temp, diff1_temp5_a2.T))
                grad_b1b3 += beta * (np.dot(diff0_temp, diff0_temp5.T) + np.dot(diff1_temp, diff1_temp5.T))                                                       
                grad_b1W4 += beta * (diff0_temp4_temp7_gradz3 * e0 + np.dot(diff0_temp, diff0_temp4_a3.T) + 
                                     diff1_temp4_temp7_gradz3 * e1 + np.dot(diff1_temp, diff1_temp4_a3.T))                       
                grad_b1b4 += beta * (np.dot(diff0_temp, diff0_temp4.T) + np.dot(diff1_temp, diff1_temp4.T))                                                       
                grad_b1W5 += beta * (diff0_temp8_gradz4 * e0 + np.dot(diff0_temp, diff0_a4.T) + 
                                     diff1_temp8_gradz4 * e1 + np.dot(diff1_temp, diff1_a4.T))   
                #------------

                #------------
                diff0_temp5_gradz2_a1 = np.kron(temp5_gradz2, a1_i) - np.kron(np.kron(temp50.T, grad_diag_z20), a1_i0)
                diff0_temp4_temp3_gradz3_a1 = np.kron(temp4_temp3_gradz3, a1_i) - np.kron(np.kron(temp40.T, np.dot(temp30, grad_diag_z30)), a1_i0)
                diff0_temp9_gradz4_a1 = np.kron(temp9_gradz4, a1_i) - np.kron(np.dot(temp90, grad_diag_z40), a1_i0)

                diff1_temp5_gradz2_a1 = np.kron(temp5_gradz2, a1_i) - np.kron(np.kron(temp51.T, grad_diag_z21), a1_i1)
                diff1_temp4_temp3_gradz3_a1 = np.kron(temp4_temp3_gradz3, a1_i) - np.kron(np.kron(temp41.T, np.dot(temp31, grad_diag_z31)), a1_i1)
                diff1_temp9_gradz4_a1 = np.kron(temp9_gradz4, a1_i) - np.kron(np.dot(temp91, grad_diag_z41), a1_i1)
                
                grad_W2W2 += beta * (np.dot(diff0_temp6_a1, diff0_temp6_a1.T) + np.dot(diff1_temp6_a1, diff1_temp6_a1.T))                               
                grad_W2b2 += beta * (np.dot(diff0_temp6_a1, diff0_temp6.T) + np.dot(diff1_temp6_a1, diff1_temp6.T))                                                 
                grad_W2W3 += beta * (diff0_temp5_gradz2_a1 * e0 + np.dot(diff0_temp6_a1, diff0_temp5_a2.T) + 
                                     diff1_temp5_gradz2_a1 * e1 + np.dot(diff1_temp6_a1, diff1_temp5_a2.T))
                grad_W2b3 += beta * (np.dot(diff0_temp6_a1, diff0_temp5.T) + np.dot(diff1_temp6_a1, diff1_temp5.T))                                                 
                grad_W2W4 += beta * (diff0_temp4_temp3_gradz3_a1 * e0 + np.dot(diff0_temp6_a1, diff0_temp4_a3.T) + 
                                     diff1_temp4_temp3_gradz3_a1 * e1 + np.dot(diff1_temp6_a1, diff1_temp4_a3.T))                       
                grad_W2b4 += beta * (np.dot(diff0_temp6_a1, diff0_temp4.T) + np.dot(diff1_temp6_a1, diff1_temp4.T))                                                 
                grad_W2W5 += beta * (diff0_temp9_gradz4_a1 * e0 + np.dot(diff0_temp6_a1, diff0_a4.T) + 
                                     diff1_temp9_gradz4_a1 * e1 + np.dot(diff1_temp6_a1, diff1_a4.T))
                #------------

                #------------
                diff0_temp5_gradz2 = temp5_gradz2 - np.kron(temp50.T, grad_diag_z20)
                diff0_temp4_temp3_gradz3 = temp4_temp3_gradz3 - np.kron(temp40.T, np.dot(temp30, grad_diag_z30))
                diff0_temp9_gradz4 = temp9_gradz4 - np.dot(temp90, grad_diag_z40)   

                diff1_temp5_gradz2 = temp5_gradz2 - np.kron(temp51.T, grad_diag_z21)
                diff1_temp4_temp3_gradz3 = temp4_temp3_gradz3 - np.kron(temp41.T, np.dot(temp31, grad_diag_z31))
                diff1_temp9_gradz4 = temp9_gradz4 - np.dot(temp91, grad_diag_z41)   

                grad_b2b2 += beta * (np.dot(diff0_temp6, diff0_temp6.T) + np.dot(diff1_temp6, diff1_temp6.T))     
                grad_b2W3 += beta * (diff0_temp5_gradz2 * e0 + np.dot(diff0_temp6, diff0_temp5_a2.T) + 
                                     diff1_temp5_gradz2 * e1 + np.dot(diff1_temp6, diff1_temp5_a2.T))
                grad_b2b3 += beta * (np.dot(diff0_temp6, diff0_temp5.T) + np.dot(diff1_temp6, diff1_temp5.T))                                                      
                grad_b2W4 += beta * (diff0_temp4_temp3_gradz3 * e0 + np.dot(diff0_temp6, diff0_temp4_a3.T) + 
                                     diff1_temp4_temp3_gradz3 * e1 + np.dot(diff1_temp6, diff1_temp4_a3.T))                       
                grad_b2b4 += beta * (np.dot(diff0_temp6, diff0_temp4.T) + np.dot(diff1_temp6, diff1_temp4.T))                                                       
                grad_b2W5 += beta * (diff0_temp9_gradz4 * e0 + np.dot(diff0_temp6, diff0_a4.T) + 
                                     diff1_temp9_gradz4 * e1 + np.dot(diff1_temp6, diff1_a4.T))
                #------------

                #------------
                diff0_temp4_gradz3_a2 = np.kron(temp4_gradz3, a2_i) - np.kron(np.kron(temp40.T, grad_diag_z30), a2_i0)
                diff0_temp3_gradz4_a2 = np.kron(temp3_gradz4, a2_i) - np.kron(np.dot(temp30, grad_diag_z40), a2_i0)

                diff1_temp4_gradz3_a2 = np.kron(temp4_gradz3, a2_i) - np.kron(np.kron(temp41.T, grad_diag_z31), a2_i1)
                diff1_temp3_gradz4_a2 = np.kron(temp3_gradz4, a2_i) - np.kron(np.dot(temp31, grad_diag_z41), a2_i1)

                grad_W3W3 += beta * (np.dot(diff0_temp5_a2, diff0_temp5_a2.T) + np.dot(diff1_temp5_a2, diff1_temp5_a2.T))
                grad_W3b3 += beta * (np.dot(diff0_temp5_a2, diff0_temp5.T) + np.dot(diff1_temp5_a2, diff1_temp5.T))                                                 
                grad_W3W4 += beta * (diff0_temp4_gradz3_a2 * e0 + np.dot(diff0_temp5_a2, diff0_temp4_a3.T) + 
                                     diff1_temp4_gradz3_a2 * e1 + np.dot(diff1_temp5_a2, diff1_temp4_a3.T))                       
                grad_W3b4 += beta * (np.dot(diff0_temp5_a2, diff0_temp4.T) + np.dot(diff1_temp5_a2, diff1_temp4.T))                                                 
                grad_W3W5 += beta * (diff0_temp3_gradz4_a2 * e0 + np.dot(diff0_temp5_a2, diff0_a4.T) + 
                                     diff1_temp3_gradz4_a2 * e1 + np.dot(diff1_temp5_a2, diff0_a4.T))
                #------------

                #------------
                diff0_temp4_gradz3 = temp4_gradz3 - np.kron(temp40.T, grad_diag_z30)
                diff0_temp3_gradz4 = temp3_gradz4 - np.dot(temp30, grad_diag_z40)

                diff1_temp4_gradz3 = temp4_gradz3 - np.kron(temp41.T, grad_diag_z31)
                diff1_temp3_gradz4 = temp3_gradz4 - np.dot(temp31, grad_diag_z41)

                grad_b3b3 += beta * (np.dot(diff0_temp5, diff0_temp5.T) + np.dot(diff1_temp5, diff1_temp5.T))                                                      
                grad_b3W4 += beta * (diff0_temp4_gradz3 * e0 + np.dot(diff0_temp5, diff0_temp4_a3.T) +
                                     diff1_temp4_gradz3 * e1 + np.dot(diff1_temp5, diff1_temp4_a3.T))                       
                grad_b3b4 += beta * (np.dot(diff0_temp5, diff0_temp4.T) + np.dot(diff1_temp5, diff1_temp4.T))                                                       
                grad_b3W5 += beta * (diff0_temp3_gradz4 * e0 + np.dot(diff0_temp5, diff0_a4.T) + 
                                     diff1_temp3_gradz4 * e1 + np.dot(diff1_temp5, diff1_a4.T))
                #------------

                #------------
                diff0_gradz4_a3 = np.kron(grad_diag_z4, a3_i) - np.kron(grad_diag_z40, a3_i0)

                diff1_gradz4_a3 = np.kron(grad_diag_z4, a3_i) - np.kron(grad_diag_z41, a3_i1)

                grad_W4W4 += beta * (np.dot(diff0_temp4_a3, diff0_temp4_a3.T) + np.dot(diff1_temp4_a3, diff1_temp4_a3.T))                      
                grad_W4b4 += beta * (np.dot(diff0_temp4_a3, diff0_temp4.T) + np.dot(diff1_temp4_a3, diff1_temp4.T))                                                 
                grad_W4W5 += beta * (diff0_gradz4_a3 * e0 + np.dot(diff0_temp4_a3, diff0_a4.T) + 
                                     diff1_gradz4_a3 * e1 + np.dot(diff1_temp4_a3, diff1_a4.T))
                #------------

                #------------
                diff0_gradz4 = grad_diag_z4 - grad_diag_z40

                diff1_gradz4 = grad_diag_z4 - grad_diag_z41
                     
                grad_b4b4 += beta * (np.dot(diff0_temp4, diff0_temp4.T) + np.dot(diff1_temp4, diff1_temp4.T))                                                       
                grad_b4W5 += beta * (diff0_gradz4 * e0 + np.dot(diff0_temp4, diff0_a4.T) + 
                                     diff1_gradz4 * e1 + np.dot(diff1_temp4, diff1_a4.T))
                #------------

                #------------
                grad_W5W5 += beta * (np.dot(diff0_a4, diff0_a4.T) + np.dot(diff1_a4, diff1_a4.T))
                #------------

        
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
