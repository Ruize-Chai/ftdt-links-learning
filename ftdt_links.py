from __future__ import annotations
import numpy as np
import manim
from scipy import constants as cst
import json

'''
FTDT-LINKS
A PROJECT FOR FTDT CALCULATION,
IMITATING MathWorks®Simulink
ONLY FOR STUDY (OR FUN)
'''

#ANSI
RESET = "\033[0m"
BOLD = "\033[1m"

RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
CYAN = "\033[36m"

BOLD_RED = "\033[1;31m"
BOLD_GREEN = "\033[1;32m"
BOLD_BLUE = "\033[1;34m"
#ANSI

class Medium:
    def __init__(self,
                 eps_x_r:np.float64,eps_y_r:np.float64,eps_z_r:np.float64,
                 mu_x_r:np.float64,mu_y_r:np.float64,mu_z_r:np.float64,
                 sigma_x:np.float64,sigma_y:np.float64,sigma_z:np.float64):
        #9 PARAMETERS in total
        self.eps_x = eps_x_r * cst.epsilon_0
        self.eps_y = eps_y_r * cst.epsilon_0
        self.eps_z = eps_z_r * cst.epsilon_0

        self.mu_x = mu_x_r * cst.mu_0
        self.mu_y = mu_y_r * cst.mu_0
        self.mu_z = mu_z_r * cst.mu_0

        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.sigma_z = sigma_z

class Param_Space:
    def __init__(self,
                 rangeX:int,
                 rangeY:int,
                 rangeZ:int):
        self.rangeX = rangeX
        self.rangeY = rangeY
        self.rangeZ = rangeZ
        self.values = np.zeros((9,
                                2*rangeX+1,
                                2*rangeY+1,
                                2*rangeZ+1),dtype = np.float64)
    
    def set_medium(self,
                   x_0:int,
                   x_1:int,
                   y_0:int,
                   y_1:int,
                   z_0:int,
                   z_1:int,
                   med:Medium
                   ):
        is_valid = True
        
        
        #范围合法性检查
        
        if(x_0>x_1 or
           y_0>y_1 or
           z_0>z_1):
            print("Invalid Range Error:Order Error")
            is_valid= False


        elif(x_0>self.rangeX or x_0<-self.rangeX or
            x_1>self.rangeX or x_1<-self.rangeX or 
            y_0>self.rangeY or y_0<-self.rangeY or 
            y_1>self.rangeY or y_1<-self.rangeY or
            z_0>self.rangeZ or z_0<-self.rangeZ or
            z_1>self.rangeZ or z_1<-self.rangeZ 
             ):
                print("Invalid Range Error:Out of Range")
                is_valid = False

        if is_valid:
        #coordinates to index conversion

            x_0 = self.rangeX + x_0 +1
            x_1 = self.rangeX + x_1 +1
            y_0 = self.rangeY + y_0 +1
            y_1 = self.rangeY + y_1 +1
            z_0 = self.rangeZ + z_0 +1
            z_1 = self.rangeZ + z_1 +1
        
        #Set Parameters
            self.values[0,x_0:x_1+1,y_0:y_1+1,z_0:z_1+1] = med.eps_x
            self.values[1,x_0:x_1+1,y_0:y_1+1,z_0:z_1+1] = med.eps_y
            self.values[2,x_0:x_1+1,y_0:y_1+1,z_0:z_1+1] = med.eps_z
            self.values[3,x_0:x_1+1,y_0:y_1+1,z_0:z_1+1] = med.mu_x
            self.values[4,x_0:x_1+1,y_0:y_1+1,z_0:z_1+1] = med.mu_y
            self.values[5,x_0:x_1+1,y_0:y_1+1,z_0:z_1+1] = med.mu_z
            self.values[6,x_0:x_1+1,y_0:y_1+1,z_0:z_1+1] = med.sigma_x
            self.values[7,x_0:x_1+1,y_0:y_1+1,z_0:z_1+1] = med.sigma_y
            self.values[8,x_0:x_1+1,y_0:y_1+1,z_0:z_1+1] = med.sigma_z


class Vector_Field:
     def __init__(self,
                  rangeX,rangeY,rangeZ):
        self.rangeX  = rangeX
        self.rangeY  = rangeY
        self.rangeZ  = rangeZ
        self.values = np.zeros((3,
                                2*rangeX+1,
                                2*rangeY+1,
                                2*rangeZ+1),dtype = np.float64)
        
     #template for partial:
     #partial A_i/partial j
     def i_partial_j(self,i,j):
         return np.gradient(self.values[i],axis=j)
     
     #partial A_x/partial j
     def x_partial_x(self):
         return self.i_partial_j(0,1)
     def x_partial_y(self):
         return self.i_partial_j(0,2)
     def x_partial_z(self):
         return self.i_partial_j(0,3)
     
     #partial A_y/partial j
     def y_partial_x(self):
         return self.i_partial_j(1,1)
     def y_partial_y(self):
         return self.i_partial_j(1,2)
     def y_partial_z(self):
         return self.i_partial_j(1,3)
     
     #partial A_z/partial j
     def z_partial_x(self):
         return self.i_partial_j(2,1)
     def z_partial_y(self):
         return self.i_partial_j(2,2)
     def z_partial_z(self):
         return self.i_partial_j(2,3)

     def bind_param_space(self,param_space:Param_Space):
        self.Param_Space = param_space
        pass
        




class Electric_Field(Vector_Field):
    def __init__(self,rangeX,rangeY,rangeZ,time_step,grid_step):
        super().__init__(rangeX,rangeY,rangeZ)
        self.h = time_step
        self.d = grid_step
        
    #绑定更新依赖场
    def bind_magnetic_field(self,H:Magnetic_Field):
        self.H = H
        

    def bind_current_field(self,J:Current_Field):
        self.J = J
        

    def update(self):
        self.values[0] = (self.values[0] + 
                          (self.h/self.d)*(self.H.z_partial_y()-self.H.y_partial_z()-self.J.values[0])/(self.Param_Space.values[0])
                          )
        self.values[1] = (self.values[1] + 
                          (self.h/self.d)*(self.H.x_partial_z()-self.H.z_partial_x()-self.J.values[1])/(self.Param_Space.values[1])
                          )
        self.values[2] = (self.values[2] + 
                          (self.h/self.d)*(self.H.y_partial_x()-self.H.x_partial_y()-self.J.values[2])/(self.Param_Space.values[2])
                          )
        



class Magnetic_Field(Vector_Field):
    def __init__(self,rangeX,rangeY,rangeZ,time_step,grid_step):
        super().__init__(rangeX,rangeY,rangeZ)
        self.h = time_step
        self.d = grid_step
        
    #绑定更新依赖场
    def bind_electric_field(self,E:Electric_Field):
        self.E = E
        
    

    def bind_current_field(self,J:Current_Field):
        self.J = J

    def update(self):
        self.values[0] = (self.values[0] -
                          (self.h/self.d)*(self.E.z_partial_y()-self.E.y_partial_z())/(self.Param_Space.values[0]))
        self.values[1] = (self.values[1] -
                          (self.h/self.d)*(self.E.x_partial_z()-self.E.z_partial_x())/(self.Param_Space.values[1]))
        self.values[2] = (self.values[2] -
                          (self.h/self.d)*(self.E.y_partial_x()-self.E.x_partial_y())/(self.Param_Space.values[2]))
        

class Current_Field(Vector_Field):

    def __init__(self,rangeX,rangeY,rangeZ):
        super().__init__(rangeX,rangeY,rangeZ)
        
    #绑定更新依赖场
    def bind_electric_field(self,E:Electric_Field):
        self.E = E
        

    def update(self):
        self.values[0] = self.values[0] + self.Param_Space.values[6]*self.E.values[0]
        self.values[1] = self.values[1] + self.Param_Space.values[7]*self.E.values[1]
        self.values[2] = self.values[2] + self.Param_Space.values[8]*self.E.values[2]


#ABOVE:BASIC ELECTROMEGNETICS EQUATION

class JSON:
    def __init__(self,json_address:str='./database.json'):
        self.json_address = json_address
        f = open(json_address,'r',encoding = 'utf-8')
        self.json_dict = json.load(f)
        


    def DUMP(self):
        f = open(self.json_address,'w',encoding = 'utf-8')
        json_str = json.dumps(self.json_dict)
        f.write(json_str)


def medium_data_load (json_class:JSON,index:int)->Medium:
    med = Medium(json_class.json_dict["medium"][index]['esp_r'][0],
                 json_class.json_dict["medium"][index]['esp_r'][1],
                 json_class.json_dict["medium"][index]['esp_r'][2],
                 json_class.json_dict["medium"][index]['mu_r'][0],
                 json_class.json_dict["medium"][index]['mu_r'][1],
                 json_class.json_dict["medium"][index]['mu_r'][2],
                 json_class.json_dict["medium"][index]['sigma'][0],
                 json_class.json_dict["medium"][index]['sigma'][1],
                 json_class.json_dict["medium"][index]['sigma'][2])
    return med

#ABOVE:JSON OPERATION







#main source

if __name__ == '__main__':
    mode_id = int(input(f"{CYAN}Please choose the mode:\n{YELLOW}1 - medium data configuration\n2 - solver main\nmode:{RESET}"))
    #BRANCH 1:DATABASE FOR MEDIUM
    if (mode_id == 1):
        print(f"{GREEN}MODE 1 START: DATABASE FOR MEDIUMS{RESET}")
        #json initial:
        Json = JSON()
        #json initial finished;
        while 1:
            J_mode_id = int(input(f"{CYAN}Please choose the mode:{YELLOW}\n1 - ADD MODE\n2 - SHOW MODE \n3 - DELETE MODE:{RESET}"))

            if J_mode_id == 1:
                #start
                #input by order
                
                #name
                name = input("name of new medium:")
                #epsilon_r x y z
                esp_r = []
                esp_r.append( np.float64( input("esp_r_x of new medium:") ) )
                esp_r.append( np.float64( input("esp_r_y of new medium:") ) )
                esp_r.append( np.float64( input("esp_r_z of new medium:") ) )

                #mu_r x y z
                mu_r = []
                mu_r.append(np.float64( input("mu_r_x of new medium:")))
                mu_r.append(np.float64( input("mu_r_y of new medium:")))
                mu_r.append(np.float64( input("mu_r_z of new medium:")))

                #sigma x y z
                sigma = []
                sigma.append(np.float64(input("sigma_x of new medium:")))
                sigma.append(np.float64(input("sigma_y of new medium:")))
                sigma.append(np.float64(input("sigma_z of new medium:")))

                new_medium_dict = {"esp_r":esp_r,
                                   "mu_r":mu_r,
                                   "sigma":sigma,
                                   "name":name}

                #add to the JSON
                Json.json_dict["medium"].append(new_medium_dict)
                #DUMP
                Json.DUMP()
                #finish

            elif J_mode_id == 2:
                i = int(0);
                for med in Json.json_dict['medium']:
                    
                    print(f"{CYAN}====name:{med["name"]}===={RESET}")
                    print(f"index : {i}")
                    print(f"esp_r_x :{med["esp_r"][0]}")
                    print(f"esp_r_y :{med["esp_r"][1]}")
                    print(f"esp_r_z :{med["esp_r"][2]}") 
                    print(f"mu_r_x :{med["mu_r"][0]}")
                    print(f"mu_r_y :{med["mu_r"][1]}")
                    print(f"mu_r_z :{med["mu_r"][2]}")
                    print(f"sigma_x :{med["sigma"][0]}")
                    print(f"sigma_y :{med["sigma"][1]}")
                    print(f"sigma_z :{med["sigma"][2]}")

                    i += 1
                    is_next = input(f"{CYAN}NEXT?[Y/n or others]:{RESET}")
                    #是否翻页到下一个Medium
                    if is_next == 'Y':
                        continue
                    else:
                        break


            elif J_mode_id == 3:
                i = int(input(f"{YELLOW}The index of medium you want to delete:{RESET}"))
                Json.json_dict['medium'].pop(i)
                pass

            else:
                print(f"{RED}MODE ERROR:invalid mode")
                exit(0)

            


    #BRANCH 2:ELECTROMAGNETICS SOLVER
    elif (mode_id == 2):
        print(f"{GREEN}MODE 2 START: ELECTROMAGNETIC FIELD SOLVER")
        pass



    else:
        print(f"{RED}MODE ERROR:invalid mode")
        exit(0)


