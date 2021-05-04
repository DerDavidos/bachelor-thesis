close all;  clear all;  clc;

%% --- Variables
%1 = MATLAB-Generator (eigen), 2 = MATLAB-Daten (Simulation), 3 = Messungen (EDF)
loadDataType = 2;

enLFP = 1;
fFIR = [100 5e3];

nStart = 12;
nFIR = 501;
nFrame = 64;
ThresSDA_min = 3e-10;


TMSpike = 0;


%% --- Interface zum Laden eines Datensatzen
cd Funktionen;
pbar = ProgressBar(21, "Verarbeitung");
switch(loadDataType)
    case 1
        file_name = '../0_Rohdaten/20201130_RawData.mat';
        load(file_name);
        SampleRate = HW_Props.ADU_SampleRate;
        GaindB = HW_Props.PreAmp_GaindB;
    case 2
        file_name = '../1_SimDaten/simulation_4.mat';
        load(file_name);
        SampleRate = 24e3;
        GaindB = 0;
        U_EL = data;
        clear data;
    case 3
        file_name = '../2_MessDaten/NP_2010.02.12_10.edf';
        RawData = edfread(file_name);
        noCh = 1;
        U_EL = transpose(cell2mat(table2array(RawData(:,noCh))));
        
        SampleRate = 20e3;
        GaindB = 0;
        clear RawData noCh;
end
pbar.update(1, "Daten eingelesen");
t0 = (1:1:length(U_EL))/SampleRate;
Uin = U_EL/(10^(GaindB/20));

%% ---- Berechnung der lokalen Feldpotentiale (LFP)
U_LFP = 0*t0;
f0 = 1.6;
U0 = 1e-3;
nLFP = 10;
for i=1:1:nLFP
    U_LFP = U_LFP + (nLFP-i-1)/nLFP* sin(2*pi*i*f0*t0);
end
Uin = Uin + enLFP*U0.* U_LFP./max(U_LFP);
clear U_LFP f0 U0 nLFP;

%% --- Spike Sorting
SDA = SpikeDetection(ThresSDA_min);
memInput = zeros(1, nFIR);
memFIR = zeros(1, nFIR);
memSpike = zeros(1, nFrame);

% --- FIR-Koeffizienten berechnen
k = 2*fFIR/SampleRate;
hFIR = fir1(nFIR-1, k, 'bandpass');
clear k;

cntSDA = 0;
cntFrame = 0;
doFrame = 0;
doFE = 0;
%SpikeFrame = {};
SpikeFrame = [];
noFrame = 0;

normSpikes = [];
pos = [];
coef_pos = [];
corrcoeff = [];
red = [];
alignSpikes = [];
alignSpikes_TypeA = [];
alignSpikes_TypeB = [];
FESpikes = [];
FESpikes_neg = [];
cnt = 1;
cnt2 = 1;
cntPos = 1;
cntNeg = 1;
pbar.update(2, "SpikeSorting (Vorbeitung done)");
nbar = 0;

% --- Algorithmus zum Spike Sorting
for i = 1:1:length(t0)
    % FIFO-Speicher (Rohdaten, FIR)
    memInput =  [memInput(2:nFIR), Uin(i)];
    U_FIR(i) = FIR(memInput, hFIR);
    memFIR = [memFIR(2:nFIR), U_FIR(i)];
    memSpike = [memSpike(2:nFrame), U_FIR(i)];

    % --- Ausführung der Spike Detection mit gleitendem Mittelwert
    SDAout(i) = SDA.performSDA(U_FIR(i));
    SpikeOut(i) = SDA.SpikeOut;
    Thres(i) = SDA.Thres;  
    
    % --- Generieren eines Spike-Frames (Schnippsel)
    % Prüfung, ob SDA nicht durch Rauschen ausgelöst wurde
    if(SDAout(i) && doFrame == 0)
        cntSDA = cntSDA + 1;
        if(cntSDA >= 1)
            cntSDA = 0;
            doFrame = 1;
        end
    end
    
    % Erstellen des Spike Frames
    if(doFrame)
        if(cntFrame >= (nFrame-nStart)) 
            noFrame = noFrame + 1;
            %SpikeFrame{noFrame} = memSpike;
            SpikeFrame(noFrame,:) = memSpike;
            cntFrame = 0;
            doFrame = 0;
            doFE = 1;
        else
            cntFrame = cntFrame + 1;
        end
    end
    
    % --- Merkmals-Extraktion auf SpikeFrame
    if(doFE)
        Spikes = Alignment(SpikeFrame,normSpikes,noFrame,pos,red,alignSpikes,alignSpikes_TypeA,alignSpikes_TypeB,cntPos,cntNeg);
        alignSpikes = Spikes.alignSpikes;
        alignSpikes_TypeA = Spikes.alignSpikes_TypeA;
        alignSpikes_TypeB = Spikes.alignSpikes_TypeB;
        normSpikes = Spikes.normSpikes;
        pos = Spikes.pos;
        cntPos = Spikes.cntPos;
        cntNeg = Spikes.cntNeg;
        SpikeFrame = Spikes.Spike;
        
        %FE = FeatureExtraction(typeFE, alignSpikes, noFrame, coef_pos, corrcoeff, FESpikes, FESpikes_neg, cnt, cnt2, TMSpike);
        py.callPython.SaveFeatures(alignSpikes, file_name);
       
        doFE = 0;
    end
    
    pbar.update(1+nbar, "Daten eingelesen");
end

cd ..