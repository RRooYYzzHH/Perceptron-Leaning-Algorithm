function [ w iterations ] = perceptron_learn( data_in )
%perceptron_learn Run PLA on the input data
%   Inputs: data_in: Assumed to be a matrix with each row representing an
%                    (x,y) pair, with the x vector augmented with an
%                    initial 1, and the label (y) in the last column
%   Outputs: w: A weight vector (should linearly separate the data if it is
%               linearly separable)
%            iterations: The number of iterations the algorithm ran for
    row = size(data_in,2);
    column = size(data_in,1);
    w = zeros(1,row-1);
    iterations = 0;
    while 1
        judge = iterations;
        for num = 1:column
            y = w * data_in(num, 1:row-1)';
            if y * data_in(num,row) <= 0
                iterations = iterations + 1;
                w = w + data_in(num,row) * data_in(num, 1:row-1);
                break;
            end
        end
        if judge == iterations
            break;
        end
    end
end

