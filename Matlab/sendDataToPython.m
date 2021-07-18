function [ret, label] = sendDataToPython(sim_type, dataNo)

    close all;  clc;

    %% --- Variables
    %Dateneingang --- 1 = MATLAB-Generator (eigen), 2,3 = MATLAB-Daten (Simulation), 4 = Messungen (Seidl), 5 = Messung (CRCNS, HC1)
    Settings.loadDataType = 1;

    %Verfahren --- 1 = PCA, 2 = Cross-Correlation, 3 = Template Matching, 4 =
    %Ternäre-Gewichtung, 5 = Ternäre Gewichtung, 6 = first and second derivative extrema (FSDE)
    Settings.typeFE = 1;
    Settings.noCluster = 5;
    %Störsignal mit lokalem Feldpotential --- 0 = aus, 1 = ein
    Settings.enLFP = 0;
    Settings.checkLabeling = 0;

    Settings.fFIR = [0.2 6]* 1e3;
    % Parameter für das Fenstern beim Spike-Frame Generator
    Settings.nStart = 0;
    Settings.nFrameSpike = 60;
    Settings.nStartAlign = 16;
    Settings.nFrameAlign = 48;
    Settings.nSNR = 80;

    % Parameter für Filterung (0: keine, 1: FIR, 2: Butter-IIR, 3: Ellip-IIR, 4: Cheby-IIR Typ 1, 5: Savitzky-Golay-FIR)
    Settings.selFILT = 5;
    Settings.nFIR = 100;
    Settings.nIIR = 1;
    % Parameter beim SpikeDetection
    Settings.ThresholdSDA = 1.2e-10;

    %% --- Interface zum Laden eines Datensatzen
    cd Funktionen;
    pbar = ProgressBar(8, "Progress");
    switch(sim_type)
        case 1
            load('../0_Rohdaten/2Cluster.mat');
            SampleRate = HW_Props.ADU_SampleRate;
            GaindB = HW_Props.PreAmp_GaindB;
            Labeling = GroundTruth;
            U_EL = U_EL/GaindB;
            clear GroundTruth HW_Props;
        case 2 % dataNo <= 95
            load(['../1_SimDaten/simulation_', num2str(dataNo), '.mat']);
            load('../1_SimDaten/ground_truth.mat');
            SampleRate = 24e3;
            GaindB = 0;
            U_EL = 25e-6*data;
            Labeling = [spike_first_sample{dataNo}; spike_classes{dataNo}];
            RefSpikes = su_waveforms{dataNo};
            clear data spike_classes spike_first_sample su_waveforms;
        case 3 % dataNo <= 5   
            load(['../1_SimDaten_Neu/simulation_', num2str(dataNo), '.mat']);
            U_EL = 0.5e-6*data;
            SampleRate = 1e3/samplingInterval;
            GaindB = 0;
            Labeling = [spike_times{1}; spike_class{1}];
            clear data samplingInterval spike_times spike_class startData chan;
        case 4
            RawData = edfread('../2_MessDaten/NP_2010.02.12_10.edf');
            noCh = 1;
            U_EL = transpose(cell2mat(table2array(RawData(:,noCh))));
            SampleRate = 20e3;
            GaindB = 0;
            Labeling = [];
            clear RawData noCh;
        case 5
            dataNo = 1;
            load(['../3_CRCNS_HC1/simulation_', num2str(dataNo), '.mat']);
            SampleRate = 10e3;
            GaindB = 0;
            Labeling = [];
    end
    pbar.update(1, "Daten eingelesen");
    clear dataNo;

    %% --- Spike Sorting
    Time = (0:1:length(U_EL)-1)/SampleRate;
    if(Settings.enLFP)
        f0 = 3.2;   U0 = 0.5e-3;  nLFP = 10;
        U_LFP = 0*Time;
        for i=1:1:nLFP
            U_LFP = U_LFP + (nLFP-i-1)/nLFP* sin(2*pi*i*f0*Time);
        end
        Uin = U0/max(U_LFP)*U_LFP + U_EL;
    else
        Uin = U_EL;
    end

    % --- Signale vordeklarieren
    U_FILT = 0*Time;
    U_SDA = 0*Time;
    U_ThSDA = 0*Time + Settings.ThresholdSDA;
    Frame_SpikeIn = [];
    Frame_SpikeAlign = [];
    Frame_SNR = {};
    Frame_SpikeCluster = {};
    FeatureTable = [];
    Cluster_SpikeProp = {};
    U_Spike = zeros(Settings.noCluster, length(Time));
    FiringRate = [];

    % --- Filter-Koeffizienten bestimmen
    switch(Settings.selFILT)
        case 0 %keine Filterung
            hFILT = 0;
        case 1 %FIR
            hFILT = fir1(Settings.nFIR, 2*Settings.fFIR/SampleRate, 'bandpass');
        case 2 %IIR (Butter)
            [hFILT{2}, hFILT{1}] = butter(Settings.nIIR, 2*Settings.fFIR/SampleRate);
        case 3 %IIR (Ellip)
            [hFILT{2}, hFILT{1}] = ellip(Settings.nIIR, 3, 60, 2*Settings.fFIR/SampleRate);
        case 4 %IIR (Cheby Typ 1)
            [hFILT{2}, hFILT{1}] = cheby1(Settings.nIIR, 3, 2*Settings.fFIR/SampleRate);
        case 5 % Savitzky-Golay
            hFILT = 0;
    end

    % --- Algorithmus zum Spike Sorting
    pbar.update(2, "SpikeSorting (Vorbeitung done)");

    %% --- Offline-Ausführung des SpikeSortings
    % --- Datenfilterung (Wavelet Denoising + Filterung)
    %U0 = wdenoise(Uin, floor(log2(length(Time)))-3, 'Wavelet','db1');
    %U_denoise = wdenoise(U0, 'DenoisingMethod', 'BlockJS');
    FILT_IN = Uin;

    switch(Settings.selFILT)
        case 0
            U_FILT = FILT_IN;
        case 1
            U_FILT = filtfilt(hFILT, 1, FILT_IN);
        case 2
            U_FILT = filter(hFILT{2}, hFILT{1}, FILT_IN);
        case 3
            U_FILT = filter(hFILT{2}, hFILT{1}, FILT_IN);
        case 4
            U_FILT = filter(hFILT{2}, hFILT{1}, FILT_IN);
        case 5
            U_FILT = sgolayfilt(FILT_IN, 4, 15);
    end

    % --- Spike Detection aus Label-Datensatz
    XPks = Labeling(1,:);
    YPks = 0* XPks;

    pbar.update(3, "Spikes erkannt");

    %% --- Frame-Generator und Alignment
    for i = 1:1:length(XPks)
        Frame_SpikeIn(i,:) = U_FILT(XPks(i)-Settings.nStart:XPks(i)+Settings.nFrameSpike-Settings.nStart-1);
        SpikeIn = movmean(abs(Frame_SpikeIn(i,:)),8);
        if(i > 1)
            dX = min([floor(0.75*(XPks(i) - XPks(i-1))), Settings.nSNR]);
        else
            dX = min([XPks(i), Settings.nSNR]);
        end
        Frame_SNR0(i,:) = Uin(XPks(i)-Settings.nStart:XPks(i)+Settings.nFrameSpike-Settings.nStart-1);
        Frame_SNR1{i} = Uin(XPks(i)-dX:XPks(i));

        max_val = 0;
        max_cnt = 0;
        max_pos = 0;
        k = 1;
        % Positions-Erkennung (Betrags-Maximum)
        while(k <= length(SpikeIn))
            if(SpikeIn(k) > max_val)
                max_val = SpikeIn(k);
                max_cnt = 0;
                max_pos = k;
            else
                max_cnt = max_cnt +1;
            end
            if(max_cnt > floor(Settings.nFrameSpike/2))
                k = length(SpikeIn)+1;
            end
            k = k +1;
        end
        % Aligned-Frame Generator
        if((max_pos - Settings.nStartAlign) <= 0)
            dX = abs(max_pos-Settings.nStartAlign);
            PlaceHolder = zeros(1, dX);
            Frame_SpikeAlign(i,:) = [PlaceHolder, Frame_SpikeIn(i,1:Settings.nFrameAlign-dX)];        
        elseif((max_pos+Settings.nFrameAlign-Settings.nStartAlign-1) > Settings.nFrameAlign)
            Frame_SpikeAlign(i,:) = U_FILT(XPks(i)+max_pos-Settings.nStartAlign:XPks(i)+max_pos+Settings.nFrameAlign-Settings.nStartAlign-1);
        else
            Frame_SpikeAlign(i,:) = Frame_SpikeIn(i, max_pos-Settings.nStartAlign:max_pos+Settings.nFrameAlign-Settings.nStartAlign-1);        
        end  
    end
    clear SpikeIn max_val max_cnt max_pos k PlaceHolder dX;
    pbar.update(4, "Alignment done");
    ret = Frame_SpikeAlign * 1e4;
    label = Labeling;

end