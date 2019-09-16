using Plots, Distributions, HDF5, LinearAlgebra

#function that takes set of polynomial coefficients, and a data set, return RMSE
#first column in data set is x value
#second column in data set is y value
#coefficients is array of polynomial coefficients   
#N = data set size, no of data points
function rootMeanSquareError(coefficients, dataset)
    
        N = size(dataset)[1]
        M = size(coefficients)[1]
        meanSquareError = 0
        polynomialFit = 0
        for i = 1:N
            for j = 1:M 
                polynomialFit = coefficients[j] * (dataset[i,1]^j)
            end
            meanSquareError += 0.5 * (polynomialFit - dataset[i, 2])^2
        end
        return meanSquareError
    end
    

#regularized least squares function
#max polynomial order = size of data
function regularLeastSquares(dataset, regParameter)
    
        #create A matrix
        N = size(dataset)[1]
        M = N 
        A = zeros(N, M)
        for i = 1:N
            for j = 1:M
                A[i, j] = dataset[N, 1]^(M-1);
            end
        end
        
        F = svd(A)
        w = zeros(M, 1);
        for i = 1:M 
            w[:] += (F.S[i] * (dot(F.U[:,i], dataset[:, 2])) /( F.S[i]^2 + regParameter)) * F.V[:, i] 
        end
        return rootMeanSquareError(w, dataset)
    end

    trainingData = h5read("C:\\Users\\Luke_\\Desktop\\School\\Machine Learning\\ps2p5.h5", "training")
    validationData = h5read("C:\\Users\\Luke_\\Desktop\\School\\Machine Learning\\ps2p5.h5", "validation")
    

  
