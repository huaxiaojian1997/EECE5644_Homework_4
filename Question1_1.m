% load an image
A = imread('EECE5644_2019Fall_Homework4Questions_3096_colorPlane.jpg');
% A = imread('EECE5644_2019Fall_Homework4Questions_42049_colorBird.jpg'); 

% Divide by 255 so that all values are in the range 0 - 1
B = double(A) / 255; 

% Size of the image
[r,c,n] = size(A);
N = r*c;

% 5-dimensional normalized feature vectors: X, Y, R, G, B
x = zeros(5,N); 

% R, G, B
RGB = (reshape(B, N, 3))';
for i = 3:5
    x(i,:) = RGB(i-2,:);
end

% X, Y
for j = 1:c
    for i = 1:r
        x(1,i+(j-1)*r) = (j-1)/c;
        x(2,i+(j-1)*r) = (i-1)/r; 
    end
end
x = x';

% Display the original image 
   figure(1);
%    subplot(1, 5, 1);
   image(A) ;
   title('Original');
%% the K-Means clustering algorithm with minimum Euclidean-distance-based assignments of samples
for K = 2:5
   % the total number of interactions of K-Means to execute
   max_iters = 3;
   
   % Initialize the centroids to be random examples
   initial_centroids = kMeansInitCentroids(x, K);
   
   % Run K-Means
   [centroids, ~] = runkMeans(x, initial_centroids, max_iters);
   
   % Find closest cluster members
   idx = findClosestCentroids(x, centroids);
   
   % Reshape the recovered image into proper dimensions
   x_recovered = centroids(idx,:);
   X = zeros(r*c,3);
   for i = 3:5
   X(:,i-2) = x_recovered(:,i);
   end
   X_recovered = reshape(X, r, c, 3);

   % Display compressed image side by side
   figure(K);
%    subplot(1, 5, K);
   image(X_recovered);
   title(sprintf('K-mean clustering classification with K=%d', K));
end
x = x';
%% GMM-based clustering
delta = 1e-1; % tolerance for k-mean and EM stopping criterion
regWeight = 1e-10; % regularization parameter for covariance estimates
d = 5; % the dimensionality of samples is 5
for K = 2:5
    alpha = ones(1,K)/K;
    shuffledIndices = randperm(N);
    mu = x(:,shuffledIndices(1:K)); % pick K random samples as initial mean estimates
    [~,assignedCentroidLabels] = min(pdist2(mu',x'),[],1); % assign each sample to the nearest mean
    Sigma = zeros(5,5,K);
    for m = 1:K % use sample covariances of initial assignments as initial covariance estimates
        Sigma(:,:,m) = cov(x(:,find(assignedCentroidLabels==m))') + regWeight*eye(d,d);
    end
    Converged = 0; % Not converged at the beginning
    temp = zeros(K,N);
    SigmaNew = zeros(5,5,K);
    while ~Converged
        for l = 1:K
            temp(l,:) = repmat(alpha(l),1,N).*evalGaussian(x,mu(:,l),Sigma(:,:,l));
        end
        plgivenx = temp./sum(temp,1);
        alphaNew = mean(plgivenx,2);
        w = plgivenx./repmat(sum(plgivenx,2),1,N);
        muNew = x*w';
        for l = 1:K
            v = x-repmat(muNew(:,l),1,N);
            u = repmat(w(l,:),d,1).*v;
            SigmaNew(:,:,l) = u*v' + regWeight*eye(d,d); % adding a small regularization term
        end
        Dalpha = sum(abs(alphaNew-alpha));
        Dmu = sum(sum(abs(muNew-mu)));
        DSigma = sum(sum(abs(abs(SigmaNew-Sigma))));
        Converged = ((Dalpha+Dmu+DSigma)<delta); % Check if converged
        alpha = alphaNew; mu = muNew; Sigma = SigmaNew;
    end
    % MAP
    MAP = zeros(K,N);
    for i = 1:K
        MAP(i,:) = log(evalGaussian(x,mu(:,i),Sigma(:,:,i)))+log(alpha(i));
    end
    [map,M] = max(MAP);
    xMAP = zeros(3,N);
    AMAP = zeros(r,c,n);
    % assign mu value to x
    for k = 1:K
        xMAP(:,find(M == k)) = repmat(mu(3:5,k),1,length(find(M == k)));
    end
    for i = 1:c
        for j = 1:r
            for k = 1:3
                AMAP(j,i,k) = xMAP(k,j+(i-1)*r);
            end
        end
    end

    % Display compressed image side by side
    figure(K+4);
    image(AMAP);
    title(sprintf('GMM-based clustering and MAP classification with K=%d', K));
end

%% function
function centroids = kMeansInitCentroids(x, K)
% Randomly reorder the indices of examples
randidx = randperm(size(x, 1));
% Take the first K examples as centroids
centroids = x(randidx(1:K), :);
end

function idx = findClosestCentroids(x, centroids)
idx = zeros(size(x,1), 1);
for i=1:length(idx)
    distanse = pdist2(centroids,x(i,:));   % compute the distance(K,1)    
       [~,idx(i)]=min(distanse);           % find the minimum
end
end

function centroids = computeCentroids(x, idx, K)
for i=1:K
       centroids(i,:) =  mean( x( find(idx==i) , :) );   
end
end

function [centroids, idx] = runkMeans(x, initial_centroids, max_iters)
% Initialize values
[m , ~] = size(x);
K = size(initial_centroids, 1);
centroids = initial_centroids;
idx = zeros(m, 1);

% Run K-Means
for i=1:max_iters
    % For each example in X, assign it to the closest centroid
    idx = findClosestCentroids(x, centroids);

    % Given the memberships, compute new centroids
    centroids = computeCentroids(x, idx, K);
end
end

function g = evalGaussian(x,mu,Sigma)
% Evaluates the Gaussian pdf N(mu,Sigma) at each coumn of X
[n,N] = size(x);
C = ((2*pi)^n * det(Sigma))^(-1/2);
E = -0.5*sum((x-repmat(mu,1,N)).*(inv(Sigma)*(x-repmat(mu,1,N))),1);
g = C*exp(E);
end