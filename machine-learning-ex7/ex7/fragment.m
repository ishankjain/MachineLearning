flag=1;
Files=dir('Jogging1/img/');
fileID=fopen('Jogging1/groundtruth_rect.txt');
formatSpec = '%f';
A = fscanf(fileID,formatSpec);
fclose(fileID);
M=vec2mat(A,4);
XY=zeros(0,4);
% Sfinal=zeros(11,1);
for j=1:4
    FileNames=Files(j).name;  
    iname=(strcat('Jogging1/img/',FileNames));
    if(strcmp(iname,'Jogging1/img/.') || strcmp(iname,'Jogging1/img/..') || strcmp(iname,'Jogging1/img/.DS_Store'))
        continue;
    end
    input_image=imread(iname);
    
    while(flag)
        first_image=imread('Jogging1/img/0001.jpg');
        x=M(1,1);
        y=M(1,2);
        w=M(1,3);
        h=M(1,4);
        XY=[XY;[x y w h]];
        p=int16(w/2);
        q=int16(h/2);
        pos=10;
        neg=10;
        references=zeros(pos+neg,2);
        rng(0,'twister');
        references(1:pos,1)=10*rand(pos,1)+x-5;
        references(1:pos,2)=0*rand(pos,1)+y;
        references(pos+1:pos+neg,1)=15*rand(neg,1)+x+5;
        references(pos+1:pos+neg,2)=0*rand(neg,1)+y;
        % disp(references);
        flag=0;
    end
    w=M(j-3,3);
    h=M(j-3,4);
    figure;imshow(iname);rectangle('Position', [x,y,w,h],...
    'EdgeColor','r','LineWidth',2 );title(iname);
    pause(1);
    close all;
    % s=size(M);
    % disp(M);
    
    
    frag=10;
    fragments=zeros(frag,2);
    radius=5*rand(frag,1);
    angle=2*pi*rand(frag,1);
    fragments(:,1)=radius.*cos(angle)+x;
    fragments(:,2)=radius.*sin(angle)+y;
    k=1;
    F=zeros(size(fragments,1),1);
    for i=1:size(fragments,1)
        bb=input_image(fragments(i,1)-p:fragments(i,1)+p,fragments(i,2)-q:fragments(i,2)+q);
        %         disp(size(bb));
        temp=mat2gray(bb);
        %     temp=bb;
        %         disp(size(temp));
        LBP=extractLBPFeatures(temp,'NumNeighbors',32);
        HOG=extractHOGFeatures(temp);
        if(size(LBP,2)>=size(HOG,2))
            HOG(numel(LBP)) = 0;
        else
            LBP(numel(HOG)) = 0;
        end
        %         disp(size(LBP,2));
        %         disp(size(HOG,2));
        fvecLBP=zeros(pos+neg,size(LBP,2));
        fvecHOG=zeros(pos+neg,size(HOG,2));
        fvec=zeros(pos+neg,size(LBP,2));
        a=zeros(1,size(LBP,2));
        for l=1:size(references,1)
            bb=input_image(references(l,1)-p:references(l,1)+p,references(l,2)-q:references(l,2)+q);
            temp=mat2gray(bb);
            basic=extractLBPFeatures(temp,'NumNeighbors',32);
            basic(numel(LBP))=0;
            fvecLBP(l,1:size(LBP,2))=basic;
            basic=extractHOGFeatures(temp);
            basic(numel(LBP))=0;
            fvecHOG(l,1:size(LBP,2))=basic;
            fvec(l,1:size(LBP,2))=sqrt((LBP-fvecLBP(l,1:size(LBP,2))).^2+(HOG-fvecHOG(l,1:size(LBP,2))).^2);
            a=a+fvec(l,1:size(LBP,2));
        end
        a=a/size(references,1);
        %     disp(a);
        %     disp(mean(a));
        avg=mean(a);
        F(k,1:1)=exp((-1*avg*avg)/(2*0.2*0.2))/(0.2*sqrt(2*pi));
        %     plot(X-x,S(k,1:1),'o');
        % disp(feature);
        k=k+1;
    end
    x=sum(fragments(:,1).*F)/sum(F);
    y=sum(fragments(:,2).*F)/sum(F);
    XY=[XY;[x y w h]];
    idx=kmeans([fragments;[x y]]);
    
end

