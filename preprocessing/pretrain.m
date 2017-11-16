function w = pretrain (val)
    val = resample(val, 250, 360);
    val = transpose(val);
    val = rescale(val);
    
    a1 = (val(1:250, :))/1000;
    a2 = (val(251:500, :))/1000;
    a3 = (val(501:750, :))/1000;
    a4 = (val(751:1000, :))/1000;
    a5 = (val(1001:1250, :))/1000;
    a6 = (val(1251:1500, :))/1000;
    a7 = (val(1501:1750, :))/1000;
    a8 = (val(1751:2000, :))/1000;
    a9 = (val(2001:2250, :))/1000;
    a10 = (val(2251:2500, :))/1000;
	
    %figure(1);
    %plot(val);
    %hold on
    %plot(a1);
    
end