clear all;
%%%Read training samples
dirOutput=dir(fullfile(strcat(pwd,'\train_frame'),'*.jpg'));%frame size 720*570
dir_frame={dirOutput.name}';

train_x=[];
train_y=[];
for i=1:length(dir_frame)


fprintf('loading %d  images\n ',i);

temp=double(rgb2gray(imread(strcat(pwd,'\train_frame\',dir_frame{i}))))/255; %load training images

label_folder=dir_frame{i}(1:6);
label=load(strcat(pwd,'\train_label\',label_folder,'\',dir_frame{i}(1:end-3),'mat')); %load ground truth location
roi=load(strcat(pwd,'\train_label\',label_folder,'\roi.mat')); %load roi region
per_map=load(strcat(pwd,'\train_perspective\',label_folder,'.mat'));%load perspective map

yn_truth(i)=label.point_num;
yloc_truth{i}=label.point_position;

[x_row y_col]=size(label.point_position);
ran_point_x=randi(576,100,1);
ran_point_y=randi(720,100,1);
ran_point=[ran_point_y ran_point_x];
in=inpolygon(ran_point(:,1),ran_point(:,2),roi.maskVerticesXCoordinates,roi.maskVerticesYCoordinates);
vali_point=ran_point(find(in==1),:);

hog1=zeros(size(vali_point,1),324);

y_truth=zeros(size(vali_point,1),1);

for j=1:length(vali_point)
    height_offset=floor(per_map.pMap(vali_point(j,2),1));
    width_offset=floor(height_offset);  %assume people are eclipse,so height:width=2:1 
           
     if (vali_point(j,1)+width_offset)>720
        x_edge=720;
    else
        x_edge=vali_point(j,1)+width_offset;
    end
        if (vali_point(j,2)+height_offset)>576
        y_edge=576;
    else
        y_edge=vali_point(j,2)+height_offset;
    end
    %crop images based on perspective map

    if(~isempty(label.point_position)) 
        x=and(label.point_position(:,1)>=vali_point(j,1),label.point_position(:,1)<=vali_point(j,1)+width_offset);
        y=and(label.point_position(:,2)>=vali_point(j,2),label.point_position(:,2)<=vali_point(j,2)+height_offset);
        indx=find(and(x,y)==1);      
        if(~isempty(indx))
        y_truth(j)=1;
        end
    end
    crop_2dim_temp=temp(vali_point(j,2):y_edge,vali_point(j,1):x_edge);
    crop_2dim=imresize(crop_2dim_temp,[150 150]);
    hog1(j,:)= extractHOGFeatures(crop_2dim,'CellSize',[32 32]);
end

temp_x=train_x;
temp_y=train_y;
train_x=vertcat(hog1,temp_x);
train_y=vertcat(y_truth,temp_y);
end


disp('Begin Training');
model=fitcsvm(train_x,train_y,'KernelFunction' ,'linear');
%model=fitcknn(train_x,train_y);
disp('Training successfully');
t_x=double(rgb2gray(imread(strcat(pwd,'\100156_A02IndiaWE-03-S20100626080000000E20100626233000000_new.split.108_1.jpg'))))/255;
label_tx=load(strcat(pwd,'\train_label\100156\100156_A02IndiaWE-03-S20100626080000000E20100626233000000_new.split.108_1.mat'));

ccount=1;
ccrop_1dim=zeros(72*90,324);
for width=1:8:720  %(720=72*10)
    for height=1:8:576   %(576=72*8)
      height_offset=floor(per_map.pMap(height));
      width_offset=floor(height_offset);
     if (width+width_offset)>720
        x_edge=720;
    else
        x_edge=width+width_offset;
    end
        if (height+height_offset)>576
        y_edge=576;
    else
        y_edge=height+height_offset;
    end
        ccrop_2dim_temp=t_x(height:y_edge,width:x_edge);
        ccrop_2dim=imresize(ccrop_2dim_temp,[150 150]);
        hog2 = extractHOGFeatures(ccrop_2dim,'CellSize',[32 32]);
        ccrop_1dim(ccount,:)=hog2;
        ccount=ccount+1;
    end
end

%[hog2,visualization] = extractHOGFeatures(t_x,'CellSize',[32 32]);
p=predict(model,ccrop_1dim);

%cont=find(p==1);
imshow(t_x);
m=reshape(p,72,90);
figure;
[row col]=find(m==1);
in=inpolygon(col,row,roi.maskVerticesXCoordinates/8,roi.maskVerticesYCoordinates/8);


indexx=find(in==0);
m(row(indexx),col(indexx))=0;

%m=flipud(m);
%contour(m);
ime=imresize(m,[576 720]);
%ime=m;

H1 = fspecial('gaussian',[7 7]);
H2 = fspecial('gaussian',[15 7]);
H=[H1;H2];
%imm=filter2([0.2 0.5 0.2;0.5 1 0.5;0.2 0.5 0.2;0.1 0.2 0.1; 0.1 0.5 0.1;0.2 1 0.2;0.1 0.5 0.1;0.1 0.2 0.1],ime);
imm=filter2(H,ime);

imshow(imm);
colormap jet;
hold on;

plot(roi.maskVerticesXCoordinates,roi.maskVerticesYCoordinates);
figure;
imshow(t_x);
hold on;
plot(label_tx.point_position(:,1),label_tx.point_position(:,2),'go');

