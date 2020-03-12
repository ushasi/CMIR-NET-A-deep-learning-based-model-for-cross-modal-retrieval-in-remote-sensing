%new_feats = normc(fc_features);
%featsx = featuresx;
%featsy = featuresy;
featsx = featuresy(1001:1400,:);
featsy = featuresx(1001:1400,:);
samples = (1:50);%10001:10050 20001:20050 30001:30050 40001:40050 50001:5050 ];%101:110 201:210 301:310 401:410 501:510 601:610 701:710 801:810 901:910 1001:1010 1101:1110 1201:1210 1301:1310 1401:1410 1501:1510];
%samples = [1:80 801:880 1601:1680 2401:2480 3201:3280 4001:4080 4801:4880 5601:5680 6401:6480 7201:7280 8001:8080 8801:8880 9601:9680 10401:10480 11201:11280 12001:12080 12801:12880 13601:13680 ];
%siam_knn = zeros(30400,30400);
%siam_Ind = zeros(30400,30400);
%distance = zeros(2100,2100);   
%const = 2000;        %change
q =size(samples) 
t=300;

for i=1: q(2)      %change
    %class = mod(i,100);%-const;       
    query = samples(i);
    

    for x=1:400%2048%80000 %23833          %change
        sum=0;sum2=0;%class2=mod(x,100);
        for y=1:t
            
            sum2 = sum2 + (sqrt(featsx(x,y) -  featsy(query,y)) * (featsx(x,y) -  featsy(query,y)));
            %sum2 = sum2 + (abs(new_feats(x,y) -  new_feats(query,y)) );

            %sum2 = sum2 + mahal(new_feats(x,:),new_feats(query,:));
            
        end
        
        sum2 = sqrt(sum2);
        distance(x,samples(i)) = sum2;     %distance(x-const,class) = sum;        
    end
    [unified_knn, unified_Ind] = sort(distance, 1);      %change
    
   query
    
end
%%check cftool
% grid on;
% plot([0 recallsiam], [100 precisionsiam] )
% hold on;
% plot( [0 recallgcn], [100 precisiongcn] )
% hold on;
% plot([0 recallcnn], [100 precisioncnn] )
% hold on;
% 
% % plot([0  cnn2r], [100 cnn2p] )
% % hold on;
% % plot( [0 chistr], [100 chistp] )
% % hold on;
% % plot( [0 gcnknnr], [100 gcnknnp] )
% % hold on;
% % 
% % plot( [0 vlfr], [100 vlfp] )
%  xlabel('Recall %'); ylabel('Precision %')
%  title(['Precision-recall curve ' ])
%  axis([0 90 70 100])
%  legend('SGCN','GCN','CNN')%,'CNN','Color Hist','GCN KNN','VLFeat')
% 

% s = [0 10 20 30 40 50 60 70 80 90 100];
%  grid on;
% plot(s, [100 gcn2p] )
% hold on;
% plot( s, [100 SIAM2p] )
% hold on;
% plot(s, [100 cnn2p] )
% hold on;
% plot( s, [100 chistp] )
% hold on;
% plot( s, [100 gcnknnp] )
% hold on;
% plot( s, [100 vlfp] )
% xlabel('Recall'); ylabel('Precision')
% title(['Precision-recall curve ' ])
% axis([0 100 0 100])
% legend('GCN','Siamese','CNN','Color Hist','GCN KNN','VLFeat')


%y = [6.00 1.01; 12.40 7.88; 94.18 60.71; 99.05 64.81];% 87 92 94;64 93 97;
%56 81 79];#
%y = [6 12.40 94.18 99.05; 1.01 7.88 60.71 64.81];
%bar(y)
%grid on;
%xlabel('Groups'); ylabel('mAP')
%title(['mAP values using different losses' ])
%axis([0 3 0 100])
%legend('Without latent loss','Without decoder loss','Without classifier loss','With total loss')%,'CNN','Color Hist','GCN KNN','VLFeat')
%names = {'Pan to Mul','Image to Audio'};
%set(gca, 'xticklabels',names);

Ind = unified_Ind;
%act = labels';
for i=1:q(2)
    for j = 1:400%23833
        pred(j,i) = 1+ floor((Ind(j,i)-1)/100);
        act(j,i) = 1+ floor((i-1)/100);
    end
    i;
end


display('tantanaaan');
%pred = pred(1:100,:);
%pred=pred';
%act=act';

%% precision
for n=1:5
    tp = 0; fp = 0; preci = 0;
    for i=1:q(2)
        tp = 0; fp = 0; preci = 0;
        for j= 1: n*10
            if (pred(j,samples(i)) == act(j,samples(i)) )
                tp = tp + 1;
            else fp = fp + 1;
            end
        end
    prec(i) = tp/(tp+fp);
    sum = mean(prec);
    end
    precision (n) = sum*100;%*100/n;

    n  
end

rec= zeros(1,q(2));
%% recall


for n=1:10
    tp = 0; total = 100; reca = 0;
    for i=1:q(2)
        tp = 0;  
        for j= 1: n*10
           if (pred(j,samples(i)) == act(j,samples(i)) )
                tp = tp + 1;
            end
        end
    rec(i) = tp/total;
    sum2 = mean(rec);
    end  
    recall(n) = sum2*100 ;%/190;
    n
end

%% MAP
map = mean(precision);

%% ANMRR
