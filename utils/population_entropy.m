function ent = population_entropy(X, negMode)
%X is the data matrix (n_cells x n_timepoints).
%negMode (optional) controls how negative activity values are handled before
%the entropy computation, which requires a non-negative probability vector:
%   'clip'            (default) negatives set to zero  -> X(X<0)=0   [published behaviour]
%   'minshift_col'    subtract each timepoint's (column) minimum so the most
%                     negative neuron maps to 0; discards no data
%   'minshift_global' add the global minimum across the matrix so all values >=0
%   'abs'             use abs(X); suppression counts as activity
% The 'clip' default leaves all existing callers unchanged. The other modes
% exist for the clipping-control analysis (Analyse_EntropyClippingControl.m).
% NOTE: spatial_entropy.m is a byte-identical copy of this function and is
% kept in sync; consolidation is out of scope here.

	if nargin < 2 || isempty(negMode)
		negMode = 'clip';
	end

	%handle negative values (Shannon entropy needs a non-negative distribution)
	switch negMode
		case 'clip'
			X(X < 0) = 0; %Clips negative values to zero (can happen with noise in dF/F data)
		case 'minshift_col'
			X = bsxfun(@minus, X, min(X, [], 1)); %per-timepoint shift so column min -> 0
		case 'minshift_global'
			X = X - min(X(:)); %single global shift so all values >= 0
		case 'abs'
			X = abs(X); %suppression counts as activity
		otherwise
			error('population_entropy:badNegMode', ...
				'Unknown negMode "%s" (use clip | minshift_col | minshift_global | abs)', negMode);
	end

	ent = X + 1e-8; %Adds a tiny epsilon to avoid log(0) = -∞ errors later

	%compute population entropy
	ent = bsxfun(@rdivide, ent, sum(ent, 1)); %Normalizes each timepoint (column) to sum to 1, treating cell activities as a probability distribution.
	ent = -sum(ent .* log2(ent), 1) ; %Computes Shannon entropy
    ent = ent/log2(size(X,1));
end
