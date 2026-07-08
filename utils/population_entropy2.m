function ent = population_entropy2(X)
%X is the data matrix (n_cells x n_timepoints). 
	%prevent numerical errors
	X(X < 0) = 0;
	ent = X + 1e-8;

	%compute population entropy
	ent = bsxfun(@rdivide, ent, sum(ent, 1));
	ent = -sum(ent .* log2(ent), 1) ;
end

