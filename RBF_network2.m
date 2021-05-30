close all
clear
format compact
load('RBF_network_training_data.mat')

%procent zbioru przeznaczonego na testowanie
ratio = 0.9;
%ilość instacji w zbiorze
instancesCount = length(T);

mn_vec = 2:1:25;    %maksymalna liczba neuronow
sc_vec = 1:0.2:10;  %stała rozkładu radialnej funkcji przejścia
testCount = 20;
iter = 0;    %licznik iteracji
iteration_count = length(mn_vec)*length(sc_vec)*testCount; %ilość iteracji
PK_avg = zeros(length(mn_vec),length(sc_vec));
MSE_avg = PK_avg;

for i = 1:testCount 
    %generowanie zbioru uczacego i testujacego oraz określenie elementów, które będą do nich należeć
    [train,test] = crossvalind('Holdout',instancesCount,ratio);
    Pn_train = zeros(4,sum(train));
    Pn_test = zeros(4,sum(test));
    T_train = zeros(1,sum(train));
    T_test = zeros(1,sum(test));

    %przypisywanie odpowiednich elemtnow do zbioru uczacego i  testujacego
    Pn_train = Pn(:,train);
    Pn_test = Pn(:,test);
    T_train = T(:,train);
    T_test = T(:,test);
    
    PK              = zeros(length(mn_vec),length(sc_vec)); %macierz poprawności klasyfikacji
    MSE             = PK;   %macierz błędu średniokwadratowego
    %pętla zmieniajaca ilość neuronow
    for ind_mn = 1:length(mn_vec)
        %pętla zmieniajaca wartości stałej rozkładu
        for ind_sc = 1:length(sc_vec)
            %tworzenie i uczenie sieci
            [net,tr] = newrb(Pn_train, T_train, 0.25/length(T_train), sc_vec(ind_sc), mn_vec(ind_mn), length(train));
            %symulacja stworzonej sieci
            Ys = sim(net, Pn_test);
            %obliczanie poprawności klasyfikacji w %
            PK_temp=(1-sum(abs(T_test-Ys)>=0.5)/length(Pn_test))*100;
            %umiesczanie wartość błędu w macierzy
            MSE(ind_mn, ind_sc) = tr.perf(length(tr.perf));
            MSE_avg(ind_mn, ind_sc) = (MSE(ind_mn, ind_sc)/testCount) + MSE_avg(ind_mn, ind_sc);
            %umieszczenie wartosci PK_temp w macierzy poprawności klasyfikacji
            PK(ind_mn, ind_sc) = PK_temp;
            PK_avg(ind_mn, ind_sc) = (PK(ind_mn, ind_sc)/testCount) + PK_avg(ind_mn, ind_sc);
            iter = iter + 1; 
            %wyświetlanie postępu
            [iter/iteration_count*100 PK_temp]
        end
    end

    %tworzenie wykresu poprawnści klasyfikacji dla danej iteracji
    figure(1), mesh(sc_vec, mn_vec, PK), xlabel('sc'), ylabel('mn'), title('PK');
    %tworzenie wykresu błędu średniokwadratowego dla danej iteracji
    figure(2), mesh(sc_vec, mn_vec, MSE), xlabel('sc'), ylabel('mn'), title('MSE');
end

%tworzenie wykresów poprawności klasyfikacji i błędu na podstawie całego eksperymentu
figure(3), mesh(sc_vec, mn_vec, PK_avg), xlabel('sc'), ylabel('mn'), title('PK avg');
figure(4), mesh(sc_vec, mn_vec, MSE_avg), xlabel('sc'), ylabel('mn'), title('MSE avg');