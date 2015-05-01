clear;

load '../GCB_matrix5.mat'

A = G + C/t_step;
[i, j, val] = find(A);
triplet_mat = [i, j, val];
[numRows, numCols] = size(A);
[nnz, dummy_three] = size(triplet_mat);
matrix_dim = [numRows, numCols, nnz];
save('A.mtx', 'matrix_dim', '-ascii');
save('A.mtx', 'triplet_mat', '-ascii', '-append');



[i, j, val] = find(B);
triplet_mat = [i, j, val];
[numRows, numCols] = size(B);
[nnz, dummy_three] = size(triplet_mat);
matrix_dim = [numRows, numCols, nnz];
save('B.mtx', 'matrix_dim', '-ascii');
save('B.mtx', 'triplet_mat', '-ascii', '-append');

[i, j, val] = find(C);
triplet_mat = [i, j, val];
[numRows, numCols] = size(C);
[nnz, dummy_three] = size(triplet_mat);
matrix_dim = [numRows, numCols, nnz];
save('C.mtx', 'matrix_dim', '-ascii');
save('C.mtx', 'triplet_mat', '-ascii', '-append');



[numRows, numCols] = size(u_vec);
matrix_dim = [numCols, numRows];
save('u_vec.mtx', 'matrix_dim', '-ascii');
for i=1:numCols
    curCol = u_vec(:, i);
    save('u_vec.mtx', 'curCol', '-ascii', '-append');
end

save('t_step.mtx', 't_step', '-ascii');


