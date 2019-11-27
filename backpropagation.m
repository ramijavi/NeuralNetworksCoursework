%Initilization of variables
Input = [0,1;1,0];
DesiredOutput = [1,1];
%Weights between the input nodes and the hidden layer nodes
Weights1 = [0.3,0;0.2,0.3;0,0.2];
%Weights between the input nodes and output node
Weights2 = [0.2,-0.1];
%Weights between the hidden layer nodes and the output node
Weights3 = [-0.1,-0.2,0.1];
%Number of nodes in hidden layer
HiddenNodes = 3;
%Number of inputs
InputsNum = 2;
%Learning rate
LearnRate = 1;
%Number of epochs our algorithm will run through
Epochs = 100;
%Number of examples
Examples = 2; %TO CHANGE

for epoch = 1:Epochs
    for e = 1:Examples
        %Forward Pass
        fprintf('Example %i\n', e);
        for n = 1:HiddenNodes
            tempSum = 0;
            for i = 1:InputsNum     
                tempSum = tempSum + (Input(e,i)*Weights1(n,i));
            end
            HiddenOut(1,n) = 1.0 ./ ( 1.0 + exp(-tempSum));
        end
        fprintf('Hidden Layer Output:\n');
        fprintf('%d\n', HiddenOut);
        Output = 0;
        for n = 1:HiddenNodes
            Output = Output + (HiddenOut(n)*Weights3(n));
        end
        for n = 1:InputsNum
            Output = Output + (Input(e,n)*Weights2(n));
        end
        fprintf('Output:\n%f\n', Output);

        %Backwards Propagation
        Error = DesiredOutput(e)- Output;
        BetaOut = Output*(1-Output)*(Error);
        for n = 1:length(Weights2)
            DeltaW(1,n) = LearnRate*BetaOut*Input(e,n);
        end
        for n = 1:length(Weights3)
            DeltaW(1,length(Weights2)+n) = LearnRate*BetaOut*HiddenOut(n);            
        end
        counter = 1;
        for n = 1:HiddenNodes
           tempBeta=HiddenOut(n)*(1-HiddenOut(n))*(Weights3(n)*BetaOut);
           for i = 1:InputsNum
               if Weights1(n,i) ~= 0
                  DeltaW(1,length(Weights2)+length(Weights3)+counter) = LearnRate*tempBeta*Input(e,i); 
                  counter = counter+1;
               end
           end
        end
        fprintf('DeltaW:\n');
        fprintf('%d\n', DeltaW);
        
        %Weights Updating
        counter = 1;
        for n = 1:length(Weights2)
            Weights2(n) = Weights2(n) + DeltaW(counter);
            counter = counter+1;
        end
        for n = 1:length(Weights3)
            Weights3(n) = Weights3(n) + DeltaW(counter);
            counter = counter+1;
        end
        for n = 1:numel(Weights1)
            if Weights1(n) ~= 0
               Weights1(n) = Weights1(n) + DeltaW(counter);
               counter = counter+1; 
            end
        end
        clearvars DeltaW;
        fprintf('Epoch %3d:  Error = %f\n',epoch,Error);
    end
end
