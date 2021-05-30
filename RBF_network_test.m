close all
clear
format compact
load('RBF_network_training_data.mat')

%procent zbioru przeznaczonego na testowanie
ratio = 0.9;
%ilosc instacji w zbiorze
instancesCount = 150;
%ilosc testow
testCount = 10;

%parametry sieci
mn = 5;
sc = 9;

PK              = zeros(1,testCount); %wektor poprawnosci klasyfikacji
MSE             = PK;   %wektor bledu sredniokwadratowego

[train,test] = crossvalind('Holdout',instancesCount,ratio);
%macierz przechowywująca odpowiedzi sieci w poszczególnych testach
Ys_array = zeros(sum(test),testCount);
%macierz przechowywująca wartości targetów w poszczególnych testach
T_test_array = Ys_array;

%generowanie zbioru uczacego i testujacego
for iter = 1:testCount   %petla zmieniajaca wartosci stale rozkladu
    [train,test] = crossvalind('Holdout',instancesCount,ratio);
    Pn_train = zeros(4,sum(train));
    Pn_test = zeros(4,sum(test));
    T_train = zeros(1,sum(train));
    T_test = zeros(1,sum(test));

    %wybieranie odpowiednich elemtnow dla zbioru uczacego i  testujacego
    Pn_train = Pn(:,train);
    Pn_test = Pn(:,test);
    T_train = T(:,train);
    T_test = T(:,test);
    
    %tworzenie i trenowanie sieci
    [net,tr] = newrb(Pn_train, T_train, 0.25/length(T_train), sc, mn, length(train));
    %symulacja stworzonej sieci
    Ys = sim(net, Pn_test);
    %obliczanie poprawnosci klasyfikacji w %
    PK_temp=(1-sum(abs(T_test-Ys)>=0.5)/length(Pn_test))*100;
    %umiesczanie wartosc bledu w wektorze
    MSE(iter) = tr.perf(length(tr.perf));
    %umieszczenie wartosci PK_temp w wektorze poprawnosci klasyfikacji
    PK(iter) = PK_temp;
    %zapisanie wartości targetów w danej iteracji w macierzy
    T_test_array(:,iter) = T_test;
    %zapisanie odpowiedzi sieci w danej iteracji w macierzy
    Ys_array(:,iter) = Ys;
    [iter/testCount*100 PK_temp] %wyswietlanie postepu
end
set(gca,'xtick',0:10)
%tworzenie wykresu poprawnosci klasyfikacji w poszczególnych testach
figure(1), scatter(1:testCount, PK, 'filled','r'), xlabel('nr testu'), ylabel('PK'), title('PK');
grid on
set(gca, 'XTick', 0:testCount)
%tworzenie wykresu bledu sredniokwadratowego w poszczególnych testach
figure(2), scatter(1:testCount, MSE, 'filled','r'), xlabel('nr testu'), ylabel('MSE'), title('MSE');
grid on
set(gca, 'XTick', 0:testCount)
%obliczanie średniej poprawności klasyfikacji
avg_MSE = sum(MSE)/length(MSE)
%obliczanie średniego błędu średniokwadratowego
avg_PK = sum(PK)/length(PK)

%tworzenie wykresu poprawnosci klasyfikacji dla poszczególnych testów
for iter = 1:testCount
    figure(iter + 2),plot([1:length(T_test_array(:,iter))],T_test_array(:,iter)...
    ,'r',[1:length(Ys_array(:,iter))],Ys_array(:,iter),'g'), title(['test ' num2str(iter)]);
    grid on
    exportgraphics(figure(iter + 2),['test' num2str(iter) '_' num2str(ratio) '_ex2.png'],'Resolution',300)
end
exportgraphics(figure(1),['PK_' num2str(ratio) '_ex2.png'],'Resolution',300)
exportgraphics(figure(2),['MSE_' num2str(ratio) '_ex2.png'],'Resolution',300)