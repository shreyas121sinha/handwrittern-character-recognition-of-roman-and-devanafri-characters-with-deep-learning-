%Matlab program to test CNN for Digit Recognition
%Read image for classification
[filename, pathname]=uigetfile('*.*','Select the Input Grayscale Image');
filewithpath=strcat(pathname,filename);
I=imread(filewithpath);
figure
imshow(I)
%Classify the image using network
label=classify(net,I);
title(['Inputted handwritten character has been recognized by the cnn as : ' char(label)])
