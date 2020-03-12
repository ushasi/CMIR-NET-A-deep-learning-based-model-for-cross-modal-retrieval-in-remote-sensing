featsx = featuresx;
featsy = featuresy;
samples = [1:30]%[2,3,4,9,11];%101:110 201:210 301:310 401:410 501:510 601:610 701:710 801:810 901:910 1001:1010 1101:1110 1201:1210 1301:1310 1401:1410 1501:1510];
%samples = [1:80 801:880 1601:1680 2401:2480 3201:3280 4001:4080 4801:4880 5601:5680 6401:6480 7201:7280 8001:8080 8801:8880 9601:9680 10401:10480 11201:11280 12001:12080 12801:12880 13601:13680 ];
%siam_knn = zeros(30400,30400);
%siam_Ind = zeros(30400,30400);
%distance = zeros(2100,2100);   
%const = 2000;        %change
q =size(samples) 

for i=1: q(2)      %change
    %class = mod(i,100);%-const;       
    query = samples(i);
    

    for x=1:255%2048%80000 %23833          %change
        sum=0;sum2=0;%class2=mod(x,100);
        for y=1:128
            
            sum2 = sum2 + (sqrt(featsy(x,y) -  featsx(query,y)) * (featsy(x,y) -  featsx(query,y)));
            %sum2 = sum2 + (abs(new_feats(x,y) -  new_feats(query,y)) );

            %sum2 = sum2 + mahal(new_feats(x,:),new_feats(query,:));
            
        end
        
        sum2 = sqrt(sum2);
        distance(x,samples(i)) = sum2;     %distance(x-const,class) = sum;        
    end
    [unified_knn, unified_Ind] = sort(distance, 1);      %change
    
   query
    
end


Ind = unified_Ind;
act = labels';
act(:,18) = 1;
for i=1:q(2)
    for j = 1:255%23833
        pred(j,i) = 1+ floor((Ind(j,i)-1)/15);
        %act(j,i) = 1+ floor((i-1)/15);
    end
    i;
end


display('tantanaaan');
%pred = pred(1:1,:);

%% precision
for n=1:10
    tp = 0; fp = 0; preci = 0;
    for i=1:q(2)
        tp = 0; fp = 0; preci = 0;
        for j= 1: n*10
            if ( (act(j,pred(j,i)+1) == 1))
                tp = tp + 1;
            %else fp = fp + 1;
            end
        end
    prec(i) = tp/(j-1);
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
            if  ( (act(j,pred(j,i)+1) == 1))
                tp = tp + 1;
            end
        end
    rec(i) = tp/total;
    sum2 = mean(rec);
    end  
    recall(n) = sum2*100 ;%/190;
    n
end
map = mean(precision);