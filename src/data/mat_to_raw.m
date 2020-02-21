load('nyu_depth_v2_labeled.mat');
addpath('toolbox_nyu_depth_v2');

shp = size(images);
va = 0;
tr = 0;
te = 0;
for i = 1:shp(4)
    im = images(:,:,:,i);
    de = depths(:,:,i);

    im = imresize(imcrop(im, [7 7 626 466]), [256 256]);
    
    hm = HeatMap(de, 'Colormap',redbluecmap);
    hfig = plot(hm);
    saveas(hfig, sprintf('HeatMaps/%d.jpg', i), "jpg");
    hm = imread(sprintf('HeatMaps/%d.jpg', i));
    hm = imresize(imcrop(hm, [153 70 904 710]), [256 256]);
    
    da = cat(2, hm, im);
    rd = randi([1 6], 1, 1);
    if rd <= 4
        imwrite(da, sprintf('nyu_depths_v2/train/%d.jpg',tr));
        tr = tr + 1;
    elseif rd==5
        imwrite(da, sprintf('nyu_depths_v2/test/%d.jpg', te));
        te = te + 1;
    else
        imwrite(da, sprintf('nyu_depths_v2/val/%d.jpg', va));
        va = va + 1;
    end
    
    close all hidden
end