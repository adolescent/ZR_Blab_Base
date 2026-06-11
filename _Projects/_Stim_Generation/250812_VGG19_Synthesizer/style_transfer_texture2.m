
clear;
% pack
% net=vgg19;
[net, classes] = imagePretrainedNetwork("vgg19");
% cd F:\style
figure(4)

% for n=1:4
    % styleImage = repmat(im2double(imread(['P10' num2str(n) '.tif'])),1,1,3);
    % styleImage = repmat(im2double(imread(['style.jpg'])),1,1,3);
    styleImage = imread('style.jpg');
    % contentImage = imread('lighthouse.png');
    contentImage = imread('style.jpg');
    %     imshow(imtile({styleImage,contentImage},'BackgroundColor','w'));
    lastFeatureLayerIdx = 38;
    layers = net.Layers;
    layers = layers(1:lastFeatureLayerIdx);

    for l = 1:lastFeatureLayerIdx
        layer = layers(l);
        if isa(layer,'nnet.cnn.layer.MaxPooling2DLayer')
            layers(l) = averagePooling2dLayer(layer.PoolSize,'Stride',layer.Stride,'Name',layer.Name);
        end
    end
    lgraph = layerGraph(layers);

    %     plot(lgraph)
    %     title('Feature Extraction Network')
    dlnet = dlnetwork(lgraph);
    %% 预处理数据
    imageSize = [256,256];
    styleImg = imresize(styleImage,imageSize);
    contentImg = imresize(contentImage,imageSize);

    imgInputLayer = lgraph.Layers(1);
    meanVggNet = imgInputLayer.Mean(1,1,:);
    styleImg = rescale(single(styleImg),0,255) - meanVggNet;
    contentImg = rescale(single(contentImg),0,255) - meanVggNet;
    %% 初始化迁移图像
    noiseRatio = 0.95;
    randImage = randi([-20,20],[imageSize 3]);
    %     transferImage = noiseRatio.*randImage + (1-noiseRatio).*styleImg;
    transferImage=randImage;

    %% 定义损失函数和样式迁移参数
    styleTransferOptions.contentFeatureLayerNames = {'conv4_2'};
    styleTransferOptions.contentFeatureLayerWeights = 1;

    %     styleTransferOptions.styleFeatureLayerNames = {'conv1_1','conv2_1','conv3_1','conv4_1','conv5_1'};
    %     styleTransferOptions.styleFeatureLayerWeights = [0.5,1.0,1.5,3.0,4.0];
    %     styleTransferOptions.styleFeatureLayerNames = {'pool1'};
    %     styleTransferOptions.styleFeatureLayerWeights = [1];
    %     styleTransferOptions.styleFeatureLayerNames = {'pool1','pool2'};
    %     styleTransferOptions.styleFeatureLayerWeights = [0.5,1];
    styleTransferOptions.styleFeatureLayerNames = {'pool1','pool2','pool4'};
    styleTransferOptions.styleFeatureLayerWeights = [1,2,5];


    styleTransferOptions.alpha = 0;
    styleTransferOptions.beta = 1;


    %% 指定训练选项
    numIterations = 10000;
    learningRate =0.5;
    trailingAvg = [];
    trailingAvgSq = [];

    %% 训练网络
    dlStyle = dlarray(styleImg,'SSC');
    dlContent = dlarray(contentImg,'SSC');
    dlTransfer = dlarray(transferImage,'SSC');

    if canUseGPU
        dlContent = gpuArray(dlContent);
        dlStyle = gpuArray(dlStyle);
        dlTransfer = gpuArray(dlTransfer);
    end

    numContentFeatureLayers = numel(styleTransferOptions.contentFeatureLayerNames);
    contentFeatures = cell(1,numContentFeatureLayers);
    [contentFeatures{:}] = forward(dlnet,dlContent,'Outputs',styleTransferOptions.contentFeatureLayerNames);

    numStyleFeatureLayers = numel(styleTransferOptions.styleFeatureLayerNames);
    styleFeatures = cell(1,numStyleFeatureLayers);
    [styleFeatures{:}] = forward(dlnet,dlStyle,'Outputs',styleTransferOptions.styleFeatureLayerNames);

    %     figure

    minimumLoss = inf;

    % params=styleTransferOptions;
    for iteration = 1:numIterations
        % Evaluate the transfer image gradients and state using dlfeval and the
        % imageGradients function listed at the end of the example.
        [grad,losses] = dlfeval(@imageGradients,dlnet,dlTransfer,contentFeatures,styleFeatures,styleTransferOptions);
        [dlTransfer,trailingAvg,trailingAvgSq] = adamupdate(dlTransfer,grad,trailingAvg,trailingAvgSq,iteration,learningRate);

        if losses.totalLoss < minimumLoss
            minimumLoss = losses.totalLoss;
            dlOutput = dlTransfer;
        end

        % Display the transfer image on the first iteration and after every 50
        % iterations. The postprocessing steps are described in the "Postprocess
        % Transfer Image for Display" section of this example.
        if mod(iteration,50) == 0 || (iteration == 1)

            transferImage = gather(extractdata(dlTransfer));
            transferImage = transferImage + meanVggNet;
            transferImage = uint8(transferImage);
            transferImage = imresize(transferImage,size(styleImage,[1 2]));

            image(transferImage)
            title(['Transfer Image After Iteration ',num2str(iteration)])
            axis off image
            drawnow
        end

    end

    %% 后处理迁移图像以用于显示
    transferImage = gather(extractdata(dlOutput));
    transferImage = transferImage + meanVggNet;
    transferImage = uint8(transferImage);
    transferImage = imresize(transferImage,size(styleImage,[1 2]));
    imwrite(transferImage,['transfer10' num2str(n) '.png']);
    % imshow(imtile({contentImage,transferImage,styleImage}, ...
    %     'GridSize',[1 3],'BackgroundColor','w'));

% end



