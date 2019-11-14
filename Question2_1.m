N=1000; n = 2; K=10;
mu = zeros(n,1);
Sigma = eye(n);
Theta = [-pi,pi];
R = [2,3];
% class priors for labels -1 and 1 respectively
p = [0.35,0.65]; 
% Generate samples
label = rand(1,N) >= p(1); l = 2*(label-0.5);
 % number of samples from each class
Nc = [length(find(label==0)),length(find(label==1))];
% reserve space
x = zeros(n,N); 
% Draw samples from each class pdf
x(:,label==0) = randGaussian(Nc(1),mu,Sigma);
theta(label==1) = (-pi) + (2*pi)*rand(1,Nc(2)); % Theta(1)+(Theta(2)-Theta(1)).*rand(1,Nc(2))
r(label==1) = 2+rand(1,Nc(2)); % R(1)+(R(2)-R(1)).*rand(1,Nc(2))
x(:,label==1) = [r(label==1).*cos(theta(label==1)); r(label==1).*sin(theta(label==1))];
%  Visualize training data
figure(1)
plot(x(1,label==0),x(2,label==0),'b.'), hold on
plot(x(1,label==1),x(2,label==1),'r.'), axis equal
xlabel('x1'), ylabel('x2')
legend('class -1','class 1')
title('Training data')
%% Linear SVM
% Train a Linear kernel SVM with cross-validation
% to select hyperparameters that minimize probability 
% of error (i.e. maximize accuracy; 0-1 loss scenario)

% 10-flod cross-validation
dummy = ceil(linspace(0,N,K+1));
for k = 1:K
    indPartitionLimits(k,:) = [dummy(k)+1,dummy(k+1)]; 
end
CList = 10.^linspace(-3,7,11);
for CCounter = 1:length(CList)
    [CCounter,length(CList)],
    C = CList(CCounter);
    for k = 1:K
        indValidate = [indPartitionLimits(k,1):indPartitionLimits(k,2)];
        xValidate = x(:,indValidate); % Using folk k as validation set
        lValidate = l(indValidate);
        if k == 1
            indTrain = [indPartitionLimits(k,2)+1:N];
        elseif k == K
            indTrain = [1:indPartitionLimits(k,1)-1];
        else
            indTrain = [(1:indPartitionLimits(k-1,2)),(indPartitionLimits(k,2)+1:N)];
        end
        % using all other folds as training set
        xTrain = x(:,indTrain); lTrain = l(indTrain);
        SVMk = fitcsvm(xTrain',lTrain,'BoxConstraint',C,'KernelFunction','linear');
        dValidate = SVMk.predict(xValidate')'; % Labels of validation data using the trained SVM
        indCORRECT = find(lValidate.*dValidate == 1); 
        indINCORRECT = find(lValidate.*dValidate == -1);
        Ncorrect(k)=length(indCORRECT);
        Nincorrect(k)=length(indINCORRECT);
    end 
    PCorrect(CCounter)= sum(Ncorrect)/N; 
    PIncorrect(CCounter) = sum(Nincorrect)/N;
end

% Visualize cross-validation process
figure(2),
plot(log10(CList),PCorrect,'.',log10(CList),PCorrect,'-'),
xlabel('log_{10} C'),ylabel('K-fold Validation Accuracy Estimate'),
title('Linear-SVM Cross-Val Accuracy Estimate'), %axis equal,

% The smallest probability of error from cross-validation
[minError,indi] = min(PIncorrect);
fprintf('The smallest probability of error from cross-validation for Linear SVM is %.4f',minError);
fprintf('\n');

% Apply the Linear-SVM classifiers to the test data samples
[dummy,indi] = max(PCorrect(:)); 
[indBestC, indBestSigma] = ind2sub(size(PCorrect),indi);
CBest1= CList(indBestC); 
SVMBest = fitcsvm(x',l','BoxConstraint',CBest1,'KernelFunction','linear');
d = SVMBest.predict(x')'; % Labels of training data using the trained SVM
indINCORRECT = find(l.*d == -1); % Find training samples that are incorrectly classified by the trained SVM
indCORRECT = find(l.*d == 1); % Find training samples that are correctly classified by the trained SVM

% Visualize classification results on training data
figure(3),
plot(x(1,indCORRECT),x(2,indCORRECT),'g.'), hold on,
plot(x(1,indINCORRECT),x(2,indINCORRECT),'r.'), axis equal,
title('Training Data for Linear SVM'),
legend('correctly Classified','Incorrectly Classified'),
xlabel('x1'), ylabel('x2'), axis equal,

% The number of erroneously classified samples and the training dataset probability of error estimate
TrainingError = length(indINCORRECT);
fprintf('The number of erroneously classified samples for Linear SVM is %d',TrainingError);
fprintf('\n');
pTrainingError = length(indINCORRECT)/N; % Empirical estimate of training error probability
fprintf('The training dataset probability of error for Linear SVM is %.4f',pTrainingError);
fprintf('\n');

%% Gaussian SVM
% Train a Gaussian kernel SVM with cross-validation
% to select hyperparameters that minimize probability 
% of error (i.e. maximize accuracy; 0-1 loss scenario)

% 10-flod cross-validation
dummy = ceil(linspace(0,N,K+1));
for k = 1:K
    indPartitionLimits(k,:) = [dummy(k)+1,dummy(k+1)]; 
end
CList = 10.^linspace(-1,9,11); sigmaList = 10.^linspace(-2,3,13);
for sigmaCounter = 1:length(sigmaList)
    [sigmaCounter,length(sigmaList)],
    sigma = sigmaList(sigmaCounter);
    for CCounter = 1:length(CList)
        C = CList(CCounter);
        for k = 1:K
            indValidate = [indPartitionLimits(k,1):indPartitionLimits(k,2)];
            xValidate = x(:,indValidate); % Using folk k as validation set
            lValidate = l(indValidate);
            if k == 1
                indTrain = [indPartitionLimits(k,2)+1:N];
            elseif k == K
                indTrain = [1:indPartitionLimits(k,1)-1];
            else
                indTrain = [(1:indPartitionLimits(k-1,2)),(indPartitionLimits(k,2)+1:N)];
            end
            % using all other folds as training set
            xTrain = x(:,indTrain); lTrain = l(indTrain);
            SVMk = fitcsvm(xTrain',lTrain,'BoxConstraint',C,'KernelFunction','gaussian','KernelScale',sigma);
            dValidate = SVMk.predict(xValidate')'; % Labels of validation data using the trained SVM
            indCORRECT = find(lValidate.*dValidate == 1); 
            indINCORRECT = find(lValidate.*dValidate == -1);
            Ncorrect(k)=length(indCORRECT);
            Nincorrect(k)=length(indINCORRECT);
        end 
        PCorrect(CCounter,sigmaCounter)= sum(Ncorrect)/N;
        PIncorrect(CCounter,sigmaCounter)= sum(Nincorrect)/N;
    end 
end

% Visualize cross-validation process
figure(4),
contour(log10(CList),log10(sigmaList),PCorrect',20); xlabel('log_{10} C'), ylabel('log_{10} sigma'),
title('Gaussian-SVM Cross-Val Accuracy Estimate'), axis equal,

% The smallest probability of error from cross-validation
[minError,indi] = min(PIncorrect(:));
fprintf('The smallest probability of error from cross-validation for Gaussian SVM is %.4f',minError);
fprintf('\n');

% Apply the Gaussian-SVM classifiers to the test data samples
[dummy,indi] = max(PCorrect(:));
[indBestC, indBestSigma] = ind2sub(size(PCorrect),indi);
CBest2= CList(indBestC); sigmaBest= sigmaList(indBestSigma); 
SVMBest = fitcsvm(x',l','BoxConstraint',CBest2,'KernelFunction','gaussian','KernelScale',sigmaBest);
d = SVMBest.predict(x')'; % Labels of training data using the trained SVM
indINCORRECT = find(l.*d == -1); % Find training samples that are incorrectly classified by the trained SVM
indCORRECT = find(l.*d == 1); % Find training samples that are correctly classified by the trained SVM

% Visualize classification results on training data
figure(5),
plot(x(1,indCORRECT),x(2,indCORRECT),'g.'), hold on,
plot(x(1,indINCORRECT),x(2,indINCORRECT),'r.'), axis equal,
title('Training Data for Gaussian SVM'),
legend('correctly Classified','Incorrectly Classified'),
xlabel('x1'), ylabel('x2'), axis equal,

% The number of erroneously classified samples and the training dataset probability of error estimate
TrainingError = length(indINCORRECT);
fprintf('The number of erroneously classified samples for Gaussian SVM is %d',TrainingError);
fprintf('\n');
pTrainingError = length(indINCORRECT)/N; % Empirical estimate of training error probability
fprintf('The training dataset probability of error for Gaussian SVM is %.4f',pTrainingError);
fprintf('\n');

%% 1000 independent test samples
% Generate samples
label = rand(1,N) >= p(1); l = 2*(label-0.5);
Nc = [length(find(label==0)),length(find(label==1))]; % number of samples from each class
x = zeros(n,N); % reserve space

% Draw samples from each class pdf
x(:,label==0) = randGaussian(Nc(1),mu,Sigma);
theta(label==1) = (-pi) + (2*pi)*rand(1,Nc(2)); % Theta(1)+(Theta(2)-Theta(1)).*rand(1,Nc(2))
r(label==1) = 2+rand(1,Nc(2)); % R(1)+(R(2)-R(1)).*rand(1,Nc(2))
x(:,label==1) = [r(label==1).*cos(theta(label==1)); r(label==1).*sin(theta(label==1))];

% Apply the Linear-SVM classifiers to the test data samples
SVMBest = fitcsvm(x',l','BoxConstraint',CBest1,'KernelFunction','linear');
d = SVMBest.predict(x')'; % Labels of training data using the trained SVM
indINCORRECT = find(l.*d == -1); % Find training samples that are incorrectly classified by the trained SVM
indCORRECT = find(l.*d == 1); % Find training samples that are correctly classified by the trained SVM

% Visualize classification results on training data
figure(6)
plot(x(1,indCORRECT),x(2,indCORRECT),'g.'), hold on,
plot(x(1,indINCORRECT),x(2,indINCORRECT),'r.'), axis equal,
title('Test Data for Linear SVM'),
legend('correctly Classified','Incorrectly Classified'),
xlabel('x1'), ylabel('x2'), axis equal,

% The test dataset probability of error estimate
pTestError = length(indINCORRECT)/N; % Empirical estimate of training error probability
fprintf('The test probability of error for Linear SVM is %.4f',pTestError);
fprintf('\n');

% Apply the Gaussian-SVM classifiers to the test data samples
SVMBest = fitcsvm(x',l','BoxConstraint',CBest2,'KernelFunction','gaussian','KernelScale',sigmaBest);
d = SVMBest.predict(x')'; % Labels of training data using the trained SVM
indINCORRECT = find(l.*d == -1); % Find training samples that are incorrectly classified by the trained SVM
indCORRECT = find(l.*d == 1); % Find training samples that are correctly classified by the trained SVM

% Visualize classification results on training data
figure(7), 
plot(x(1,indCORRECT),x(2,indCORRECT),'g.'), hold on,
plot(x(1,indINCORRECT),x(2,indINCORRECT),'r.'), axis equal,
title('Test Data for Gaussian SVM'),
legend('correctly Classified','Incorrectly Classified'),
xlabel('x1'), ylabel('x2'), axis equal,

% The test dataset probability of error estimate
pTestError = length(indINCORRECT)/N; % Empirical estimate of training error probability
fprintf('The test probability of error for Gaussian SVM is %.4f',pTestError);
fprintf('\n');