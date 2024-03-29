%function createfigure(ymatrix1)
%CREATEFIGURE(YMATRIX1)
%  YMATRIX1:  bar matrix data

%  Auto-generated by MATLAB on 06-Mar-2019 20:08:43
 ymatrix1 = [99.6 99.5  ;99.9 99.8; 100 100; 99.6 99.5; 99.4 99.5;99.7 99.8; 99.9 99.9; 100 100];% 87 92 94;64 93 97; 56 81 79];

% Create figure
figure1 = figure;

% Create axes
axes1 = axes('Parent',figure1,...
    'XTickLabel',{'Aquafarm','Cloud','Forest','HighBldg','LowBldg','FarmLand','River','Water'},...
    'XTick',[1 2 3 4 5 6 7 8]);
%% Uncomment the following line to preserve the X-limits of the axes
% xlim(axes1,[0 9]);
%% Uncomment the following line to preserve the Y-limits of the axes
% ylim(axes1,[95 100]);
box(axes1,'on');
grid(axes1,'on');
hold(axes1,'on');

% Create multiple lines using matrix input to bar
bar1 = bar(ymatrix1);
set(bar1(2),'DisplayName','Pan to Mul');
set(bar1(1),'BaseValue',97,'DisplayName','Mul to Pan');

% Create xlabel
xlabel('Classes');

% Create ylabel
ylabel('mAP');

% Create title
title('Average precision of each class.');

% Create legend
legend(axes1,'show');

