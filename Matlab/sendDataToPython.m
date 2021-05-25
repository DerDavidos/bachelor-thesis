function ret = sendDataToPython(sim_type, dataNo)

    %% --- Variables
    %Dateneingang --- 1 = MATLAB-Generator (eigen), 2,3 = MATLAB-Daten
    %(Simulation), 4 = Messungen (Seidl), 5 = Messung (CRCNS, HC1)
    Settings.loadDataType = sim_type;
    %Modus des SpikeSortings --- 0 = offline, 1 = online
    Settings.mode = 0;
    %Verfahren --- 1 = PCA, 2 = Cross-Correlation, 3 = Template Matching, 4 = Ternäre-Gewichtung
    Settings.typeFE = 3;
    %Störsignal mit lokalem Feldpotential --- 0 = aus, 1 = ein
    Settings.enLFP = 0;
    Settings.checkLabeling = 0;

    Settings.fFIR = [100 6e3];
    % Parameter für das Fenstern beim Spike-Frame Generator
    Settings.nStart = 16;
    Settings.nFrameSpike = 81;
    Settings.nStartAlign = 10;
    Settings.nFrameAlign = 48;

    % Parameter für Filterung (0: keine, 1: FIR, 2: Butter-IIR, 3: Cheby-IIR)
    Settings.selFILT = 2;
    Settings.nFIR = 501;
    Settings.nIIR = 4;
    % Parameter beim SpikeDetectin
    Settings.ThresholdSDA = 1.2e-10;

    %% --- Interface zum Laden eines Datensatzen
    cd Funktionen;
    pbar = ProgressBar(21, "Verarbeitung");
    switch(Settings.loadDataType)
        case 1
            load('../0_Rohdaten/20210524_RawData.mat');
            SampleRate = HW_Props.ADU_SampleRate;
            GaindB = HW_Props.PreAmp_GaindB;
            Labeling = GroundTruth;
            clear GroundTruth HW_Props;
        case 2 % dataNo <= 95
            file = ['../1_SimDaten/simulation_', num2str(dataNo), '.mat'];
            load(file);
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
            RawData = edfread('../../2_MessDaten/NP_2010.02.12_10.edf');
            noCh = 1;
            U_EL = transpose(cell2mat(table2array(RawData(:,noCh))));
            SampleRate = 20e3;
            GaindB = 0;
            Labeling = [];
            clear RawData noCh;
        case 5
            dataNo = 1;
            load(['../../3_CRCNS_HC1/simulation_', num2str(dataNo), '.mat']);
            SampleRate = 10e3;
            GaindB = 0;
            Labeling = [];
    end
    pbar.update(1, "Daten eingelesen");
    clear dataNo;

    %% ---- Berechnung der lokalen Feldpotentiale (LFP)
    Time = (1:1:length(U_EL))/SampleRate;
    U_LFP = 0*Time;

    f0 = 3.2;
    U0 = 1e-3;
    nLFP = 10;
    for i=1:1:nLFP
        U_LFP = U_LFP + (nLFP-i-1)/nLFP* sin(2*pi*i*f0*Time);
    end
    Uin = U_EL/(10^(GaindB/20)) + Settings.enLFP* U0.* U_LFP./max(U_LFP);
    clear i U_LFP f0 U0 nLFP;

    %% --- Spike Sorting
    % --- Signale vordeklarieren
    U_FILT = 0*Time;
    U_SDA = 0*Time;
    U_ThSDA = 0*Time + Settings.ThresholdSDA;
    Frame_SpikeIn = [];
    Frame_SpikeAlign = [];

    % --- Filter-Koeffizienten bestimmen
    switch(Settings.selFILT)
        case 0 %keine Filterung
        case 1 %FIR
            hFILT = fir1(nFIR-1, 2*Settings.fFIR/SampleRate, 'bandpass');
        case 2 %IIR (Butter)
            [hFILT1, hFILT0] = butter(Settings.nIIR, 2*Settings.fFIR/SampleRate);
        case 3 %IIR (Cheby Typ 2)
            [hFILT1, hFILT0] = cheby2(Settings.nIIR, 80, 2*Settings.fFIR/SampleRate);
    end

    % --- Algorithmus zum Spike Sorting
    pbar.update(2, "SpikeSorting (Vorbeitung done)");
    if(Settings.mode == 0)
        %% --- Offline-Ausführung des SpikeSortings
        % --- Datenfilterung (Wavelet Denoising + Filterung)
        U_denoise = wdenoise(Uin, 'DenoisingMethod', 'BlockJS'); %9, 'NoiseEstimate', 'LevelIndependent');
        if(Settings.selFILT == 0)
            U_FILT = Uin;
        elseif(Settings.selFILT == 1)
            U_FILT = filter(hFILT, U_denoise);
        else
            U_FILT = filter(hFILT1, hFILT0, U_denoise);
        end

        % --- Spike Detection
        % k-TEO
        k=12;
        for i = k+1:1:length(Time)-k
            U_SDA(i) = U_denoise(i).^2 - U_denoise(i-k)*U_denoise(i+k);
        end
        U_SDA = movmean(U_SDA, k+1);
        clear k;

    %     %MTEO
    %     U_SDA1 = 0*Time;
    %     k = [1 4 7 10 13 16];
    %     U_MTEO = zeros(length(k), length(Time));
    %     for x = 1:1:length(k)
    %         for i = k(x)+1:1:length(Time)-k(x)
    %             U_MTEO(x,i) = U_denoise(i).^2 - U_denoise(i-k(x))*U_denoise(i+k(x));
    %         end
    %         U_MTEO(x,:) = movmean(U_MTEO(x,:), 2*k(x)+1);
    %     end
    %     for i = 1:1:length(Time)
    %         U_SDA1(i) = max(U_MTEO(:,i));
    %     end

        [YPks, XPks] = findpeaks(U_SDA, 'MinPeakHeight', Settings.ThresholdSDA, 'MinPeakDistance', round(500e-6*SampleRate)); 

        %% --- Frame-Generator und Alignment
        for i = 1:1:length(XPks)
            Frame_SpikeIn(i,:) = U_denoise(XPks(i)-Settings.nStart:XPks(i)+Settings.nFrameSpike-Settings.nStart-1);
            SpikeIn = movmean(abs(Frame_SpikeIn(i,:)),8);

            max_val = 0;
            max_cnt = 0;
            max_pos = 0;
            k = 1;
            % Positions-Erkennung (Betrags-Maximum)
            while(k < length(SpikeIn))
                k = k +1;
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
            end
            % Aligned-Frame Generator
            if(max_pos - Settings.nStartAlign <= 0)
                dX = abs(max_pos-Settings.nStartAlign);
                PlaceHolder = zeros(1, dX);
                Frame_SpikeAlign(i,:) = [PlaceHolder, Frame_SpikeIn(i,1:Settings.nFrameAlign-dX)];        
            else
                try               
                  Frame_SpikeAlign(i,:) = Frame_SpikeIn(i, max_pos-Settings.nStartAlign:max_pos+Settings.nFrameAlign-Settings.nStartAlign-1);
                catch
                   display(i)
                end
            end
        end
        clear SpikeIn max_val max_cnt max_pos k PlaceHolder dX;
        
        ret = Frame_SpikeAlign * 1e6;
        
    else
        %% --- Online-Ausführung des SpikeSortings
    end
end