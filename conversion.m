%Program to convert MNIST digit dataset to jpg form.
dataset= csvread('mnist_test.csv',1,0);

labels=dataset(:,1); %readlabels
digitcounter= zeros(10,1); %for 10 categories

%making folders in current directory, make different folders for training set

mkdir('mnistdigitdataset'); %main folder

%folders for each dataset of each digit

mkdir('mnistdigitdataset\0');
mkdir('mnistdigitdataset\1');
mkdir('mnistdigitdataset\2');
mkdir('mnistdigitdataset\3');
mkdir('mnistdigitdataset\4');
mkdir('mnistdigitdataset\5');
mkdir('mnistdigitdataset\6');
mkdir('mnistdigitdataset\7');
mkdir('mnistdigitdataset\8');
mkdir('mnistdigitdataset\9');

for i=1:10000
  digit= dataset(i,2:end);
  digit= uint8(reshapeobj(digit));
  getlabel= labels(i);
  
 folderpath= strcat('G:\ml_project\mnist_test.csv\mnistdigitdataset\', num2str(getlabel),'\');
 [filename,countupdate]= getfilename(getlabel, digitcounter);
 imwrite(digit, strcat(folderpath, filename));
 digitcounter=countupdate;
 end
 
  

