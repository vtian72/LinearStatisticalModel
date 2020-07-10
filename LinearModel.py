#Class to build a linear model --> only works for continuous covariates

import os
import pandas as pd
import numpy as np

class LinearModel:
    def __init__(self, dataset, dependent, independent, intercept):
        self.dataset = dataset
        self.dependent = dependent
        self.independent = independent
        self.intercept = intercept
    
    ############ Methods to set up and return required data ############ 
    def get_data(self, base_dir = "/Users/vincenttian/Documents/Programming/Data/"):
        '''Method to get read the dataset given the file is stored in the following base directory'''  
        filepath = os.path.join(base_dir, "{}.csv".format(str(self.dataset)))
        data = pd.read_csv(filepath)
        return data

    def get_response(self):
        '''Method to get the response data'''
        dataset = self.get_data()
        response = dataset.values[:,self.dependent]
        return response
    
    def get_design_matrix(self):
        '''Method to get the design matrix'''
        dataset = self.get_data()
        design_matrix = dataset.values[:,self.independent]
        #Check if an intercept is used
        if self.intercept:
            design_matrix = np.insert(design_matrix,0,1,axis=1)
        return design_matrix

    ######### Statistics of the model #########
    def get_r_squared(self):
        '''Method to return the R-squared value of the model'''
        response,rss = self.get_response(),self.get_rss()
        #total sum of squares
        response_mean = sum(response[i] for i in range(len(response)))/len(response)
        if self.intercept:
            tss = sum((response[i] - response_mean)**2 for i in range(len(response)))
        else:
            tss = sum((response[i])**2 for i in range(len(response)))
        return 1-(rss/tss)

    def get_adj_r_squared(self):
        '''Method to return the Adjusted R-squared value of the model, where:
            n = number of observations
            k = number of variables including the constant'''
        r_squared = self.get_r_squared()
        n = self.get_design_matrix().shape[0]
        k = self.get_design_matrix().shape[1]
        return 1- (((1-r_squared)*(n-1)) / (n-k)) if self.intercept else 1- (((1-r_squared)*(n-1)) / (n-k-1))

    def get_rss(self):
        '''Method to return the residual sum of squares for a model'''
        response, design_matrix, beta = self.get_response(), self.get_design_matrix(), self.get_beta()
        pred_val = design_matrix.dot(beta)
        rss = sum((response[i]-pred_val[i])**2 for i in range(len(pred_val)))
        return rss

    def get_mse(self):
        '''Method to return the mean squared error for a model'''              
        design_matrix = self.get_design_matrix()
        mse = self.get_rss()/design_matrix.shape[0]
        return mse

    def get_rse(self):
        '''Method to return the residual standard error, which is found by taking the square root of RSS/(nRows - nVar)'''
        rss = self.get_rss()
        nvar = self.get_design_matrix().shape[1]
        nrows = self.get_design_matrix().shape[0]
        return (rss/(nrows-nvar+1))**0.5

    ######### Functions to estimate model variance, evaluate beta and the standard errors of beta #########

    def get_var(self):
        '''Method to return the Variance estimation for a model'''
        response = self.get_response()
        design_matrix = self.get_design_matrix()
        beta = self.get_beta()
        pred_val = design_matrix.dot(beta)
        return sum((response[i]-pred_val[i])**2 for i in range(len(pred_val)))/(design_matrix.shape[0] - design_matrix.shape[1])

    def get_beta(self):
        '''Method to return estimates for the covariates in the model'''
        matrix_q = self.find_q().astype(np.float64)
        matrix_r = self.find_r().astype(np.float64)
        response = self.get_response().astype(np.float64)
        return (np.linalg.inv(matrix_r).dot(np.transpose(matrix_q))).dot(response)
    
    def get_beta_errors(self):
        '''Method to return the standard errors of the beta estimate'''
        matrix_r = self.find_r().astype(np.float64)
        var_estimate = self.get_var().astype(np.float64)
        beta_error_matrix = ((np.linalg.inv(matrix_r)).dot(np.transpose((np.linalg.inv(matrix_r))))*var_estimate)**0.5
        return np.diag(beta_error_matrix)

    ############## Methods to find particular matrices used in QR-Decomposition ###################
    def find_u(self):
        '''Method to find U(matrix containing the new column vectors) using the Gram-Schmidt process'''
        matrix_u = self.get_design_matrix()
        for col in range(matrix_u.shape[1]):
            if col == 0:
                continue
            else:
                vector_a = np.array(matrix_u[:,col]).reshape(-1,).tolist()
                sum_proj_u_a = [0] * matrix_u.shape[0]
                for num in range(col):
                    vector_u = np.array(matrix_u[:,num]).reshape(-1,).tolist()
                    sum_proj_u_a = [s1 + p1 for s1, p1 in zip(sum_proj_u_a,self.proj_u_a(vector_u,vector_a))]
                new_col = [a1 - s1 for a1, s1 in zip(vector_a,sum_proj_u_a)]
                matrix_u = self.place_column(matrix_u, col, new_col)
        return matrix_u

    def find_q(self):
        '''Method to find Q(orthogonal matrix) by Gram-Schmidt process.
           Calculated by dividing each value by the magnitude of its respective column from the U Matrix'''
        matrix_q = self.find_u()
        for col in range(matrix_q.shape[1]):
            col_mag = self.find_col_mag(matrix_q, col)
            matrix_q[:,col] = matrix_q[:,col]/col_mag
        return matrix_q

    def find_r(self):
        '''Method to find R (upper triangular matrix) by Gram-Schmidt process'''
        matrix_q = self.find_q()
        design_matrix = self.get_design_matrix()
        return np.transpose(matrix_q).dot(design_matrix)

    ############  Matrix Functions ############
    def dot_p(self,vector_a,vector_b):
        '''Method to find and return the dot product of 2 vectors a and b'''
        result = 0
        for index in range(len(vector_a)):
            result += vector_a[index] * vector_b[index]
        return result

    def proj_u_a(self,vector_u,vector_a):
        '''#Method to return the projection of a on u'''
        value = self.dot_p(vector_u,vector_a)/self.dot_p(vector_u,vector_u)
        return [val * value for val in vector_u]

    def find_col_mag(self,matrix, col):
        '''Method to find and return the magnitude of a column in a matrix'''
        return (sum(val**2 for val in matrix[:,col]))**0.5

    def place_column(self,matrix, col, list):
        '''#Method to place the values of a list into the respective column in a matrix'''
        matrix[:,col] = list
        return matrix
    ###########################################

#Use column 2 as response
#Use columns 3,4,5,6 as covariates
#Use an intercept
lm1 = LinearModel("kc_house_data",[2],[3,4,5],True)

print("BETA IS \n", lm1.get_beta())
print("Adjusted R-Squared is \n", lm1.get_adj_r_squared())
print("MSE is \n", lm1.get_mse())
print("RSE is \n", lm1.get_rse())
print("\nPrediction Std Error is \n", lm1.get_beta_errors())
print("\nR^2 is ", lm1.get_r_squared())
