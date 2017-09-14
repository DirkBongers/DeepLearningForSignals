%Example Link: https://physionet.org/atm/mitdb/101/atr/0/10/export/matlab/101m.mat
sampleRange = 100:234;
data = []
for i = sampleRange
    loadstr = ['https://physionet.org/atm/mitdb/',num2str(i),'/atr/0/10/export/matlab/',num2str(i),'m.mat'];
    try
        url = loadstr;
        filename = [num2str(i),'m.mat'];
        websave(filename,url);
    catch
        disp('failed')
    end
end