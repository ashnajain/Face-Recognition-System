% Face detection using eigen based approach
imagefiles = dir('*.jpg');  
nfiles = length(imagefiles);    % Number of files found
images=[];
filename =[];
colorimg=[];
filenames = [];
sketchornot = [];
for ii=1:nfiles                 %loop to resize all the images and save it in images
   currentfilename = imagefiles(ii).name;
   tf =  contains(imagefiles(ii).name,"O");
   if tf==0
       sketchornot(ii) = 1;
   else
       sketchornot(ii) = 0;
   end
   filename{ii} = imagefiles(ii).name;
   currentimage = imread(currentfilename);
   currentimage = imresize(currentimage, [20 20]);
   images{ii} = currentimage;
   colorimg{ii} = currentimage;
end
disp(sketchornot);
%resizing the matrix
X=[];
for ii=1:nfiles
   %figure;
   %imshow(images{ii});
   images{ii} = rgb2gray(images{ii});
   [m,n] = size(images{ii});
   temp = reshape(images{ii}',m*n,1); 
    X = [X temp];       
end

%calculating the mean of the marix
m = mean(X,2); 
imgcount = size(X,2);

%computing the mean subtracted matrix
A = [];
for i=1 : imgcount
    temp = double(X(:,i)) - m;
    A = [A temp];
end


L= A' * A;
[V,D]=eig(L);  
s=0;
% for i = 1 : size(V,2) 
%         s = s+D(i,i);
% end
% for i = 1 : size(V,2) 
%         if (s-D(i,i)/s)> 0.96
%             test(i,i)=1;
%         end
% end
% disp(test);
%retaining all the important eigen vectors
L_eig_vec = [];
for i = 1 : size(V,2) 
    if( D(i,i)>1 )
        L_eig_vec = [L_eig_vec V(:,i)];
    end
end

%computing the eigen faces
eigenfaces = A * L_eig_vec;
imshow(eigenfaces);
projectimg = [ ];  % projected image vector matrix
for i = 1 : size(eigenfaces,2)
    temp = eigenfaces' * A(:,i);
    projectimg = [projectimg temp];
end

%image folder contains the path to test images
imagefolder= 'D:\ch'
testimagefiles = dir(fullfile(imagefolder,'*.jpg'));
ntestfiles = length(testimagefiles);    % Number of files found
disp(ntestfiles);
testfilename = []
imagestest=[]
colorimgtest=[]
cnt = 0;

for ii=1:ntestfiles
   currenttestfilename= fullfile(imagefolder, testimagefiles(ii).name);         % it will specify images names with full path and extension
   testfilename{ii} = testimagefiles(ii).name;
   currenttestimage = imread(currenttestfilename);
   test_image = imresize(currenttestimage, [20 20]);
   test_image = test_image(:,:,1);
   [r c] = size(test_image);
   temp = reshape(test_image',r*c,1); % creating (MxN)x1 image vector from the 2D image
   temp = double(temp)-m; % mean subtracted vector
   projtestimg = eigenfaces'*temp; % projection of test image onto the facespace
 
   %computing the euclid distance
   euclide_dist = [ ];
   for i=1 : size(eigenfaces,2)
      temp = (norm(projtestimg-projectimg(:,i)))^2;
      euclide_dist = [euclide_dist temp];
   end

   %index with minimum euclid distance is retained
   [euclide_dist_min recognized_index] = min(euclide_dist);
   figure;
   imshow(colorimg{recognized_index});
   if sketchornot(recognized_index)==1
       disp("Sketch");
   else
       disp("Color");
   end
   
   tf = strncmp(testfilename{ii},filename{recognized_index},9);
   disp(filename{recognized_index});        %displaying the recognized image
   
   if tf==1     %if the two image match then cnt is incremented
       cnt = cnt+1;
   end
end
accuracy = (cnt/ntestfiles)*100;        %computing the accuracy
disp(accuracy);                         %displaying the accuracy