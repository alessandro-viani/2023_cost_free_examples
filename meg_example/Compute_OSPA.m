function OSPA = Compute_OSPA(dipoles, DIP, PERM)

cutoff = 0.0;
N1 = size(dipoles,1);
N2 = size(DIP,1);
if N1 <= N2
    X = dipoles;
    Y = DIP(1:N2,:);
else
    X = DIP(1:N2,:);
    Y = dipoles;
end
PERM_ROW = min([N1 N2]);
all_dist = zeros(size(PERM(PERM_ROW).perm,1),1);
if PERM_ROW > 0
    for i = 1:size(PERM(PERM_ROW).perm,1)
        all_dist(i) = 0;
        if max(PERM(PERM_ROW).perm(i,:)) <= size(Y,1) %%else we're
            %attempting a permutation with non existing dipoles %~(max(PERM
            %(PERM_ROW).perm(i,:)) > N2 && max(PERM(PERM_ROW).perm(i,:)) >
            %N1)
            for j = 1:size(PERM(PERM_ROW).perm,2)
                all_dist(i) = all_dist(i) + norm(X(j,1:3)-Y(PERM(PERM_ROW).perm(i,j),1:3));
            end
        else
            all_dist(i) = NaN;
        end
    end
    if N2 > 0
        MinDist = zeros(1,N1);
        for i = 1:N1
            clear newdist
            for j = 1:N2
                newdist(j) = norm(dipoles(i,1:3) - DIP(j,1:3));
            end
            %            DistForHist = [DistForHist min(newdist)];
            MinDist(i) = min(newdist);
            %STDS = [STDS; (det(ris.est(t).Sigma(:,:,i)))^(1/6)];
        end
        AveMinDist = mean(MinDist);
    end
    disp(all_dist);
    %disp(strcat(['size all_dist = ', num2str(size(all_dist))]))
    OSPA = (min(all_dist) + cutoff*(abs(N1 - N2)))/size(X,1);
else
    OSPA = -1;
    disp('Error - number of rows not compatible')
end
end
