clear PERM
for i=1:5 
    PERM(i).perm(1,:) = [1:i];
    PERM(i).indice = 1;
end
PERM(1).perm = [1;2;3;4;5];
for i = 1:5
    for j=1:5
        if j~=i
            PERM(2).perm(PERM(2).indice,:) = [i j];
            PERM(2).indice = PERM(2).indice +1;
            for k=1:5
                if k~=i && k~=j
                    PERM(3).perm(PERM(3).indice,:) = [i j k];
                    PERM(3).indice = PERM(3).indice+1;
                    for l=1:5
                        if l~=i && l~=j && l~=k
                            PERM(4).perm(PERM(4).indice,:) = [i j k l];
                            PERM(4).indice = PERM(4).indice+1;
                            for m = 1:5
                                if m~=i && m~=j && m~=k && m~=l
                                    PERM(5).perm(PERM(5).indice,:) = [i j k l m];
                                    PERM(5).indice = PERM(5).indice + 1;
                                end
                            end
                        end
                    end
                end
            end
        end
    end
end