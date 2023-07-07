function [pos, data, data_no_noise, orient] = data_creation(n_dip, sourcespace, LF, noise_std, name)

pos = zeros(n_dip,1);
pos(1) = randi(size(sourcespace,1));

for ii=2:n_dip
    pos(ii) = randi(size(sourcespace,1));
    dist = min(sqrt(sum((sourcespace(pos(1:ii-1),:) - sourcespace(pos(ii),:)).^2,2)));
    
    while dist < 0.05
        pos(ii) = randi(size(sourcespace,1));
        Compute_PERM;
        dist = min(norm(sourcespace(pos(1:ii-1),:) - sourcespace(pos(ii),:)));
    end
end

ori = [1 0 0; 0 1 0; 0 0 1];
data_size = LF(:,1:3)*ori(1, :)'*[1:100];

orient = zeros(n_dip,1);
data = zeros(size(data_size));
for i_dip = 1:n_dip
    amp = exp(-([1:100]-50).^2./2/20^2);
    G = LF(:,pos(i_dip)*3-2:pos(i_dip)*3); 
    data_1 = G*ori(1, :)'*amp;
    data_2 = G*ori(2, :)'*amp;
    data_3 = G*ori(3, :)'*amp;
    [~, idx_max_data] = max([max(max(abs(data_1))),max(max(abs(data_2))),max(max(abs(data_3)))]);
    data = data + G*ori(idx_max_data, :)'*amp;
    orient(i_dip) = idx_max_data; 
end
data_no_noise = data;
data = data + noise_std*randn(size(data));
save(strcat('data/data_', num2str(name), '.mat'), 'pos', 'data', 'data_no_noise', 'orient');
end