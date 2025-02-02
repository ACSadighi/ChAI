use Math;
use Random;

proc nll_loss(input: [] real, target: [] int, weights: [] real, ignore_index: int = -1, reduction: string = "mean"): real {
    var C = weights.size; // Number of classes
    var N = input.domain.dim(0).size; // Number of samples (rows of input)
    var total_loss = 0.0; // Total loss accumulator
    
    // Loop through each sample
    for i in 1..N {
        var target_class = target[i];
        
        // Skip the sample if its class is equal to ignore_index
        if target_class == ignore_index then
            continue;
        
        // Get the log-probability for the true class of the sample
        var log_prob = input[i, target_class];
        
        // Compute the weight for the current class
        var class_weight = weights[target_class];
        
        // Compute the individual loss
        var loss = -class_weight * log_prob;
        total_loss += loss;
    }

    // Compute the final loss based on the reduction type
    if reduction == "mean" {
        total_loss /= N; // Average the loss
    } else if reduction == "sum" {
        // No further modification, already summed
    } else {
        // If the reduction type is invalid, return 0.0
        return 0.0;
    }

    // Return the computed loss
    return total_loss;
}

proc kldiv_loss(input: [] real, target: [] real, log_target: bool = false, reduction: string = "mean"): real {
    var N = input.domain.dim(0).size; // Number of samples (rows of input)
    var total_loss = 0.0; // Total loss accumulator
    
    // Loop through each sample
    for i in 1..N {
        // Compute pointwise KL divergence
        if log_target {
            // If target is already in log-space
            total_loss += target[i] * (Math.exp(target[i] - input[i]));  // Use Math.exp()
        } else {
            // If target is in probability space
            total_loss += target[i] * (log(target[i]) - input[i]);
        }
    }

    // Compute the final loss based on the reduction type
    if reduction == "mean" {
        total_loss /= N; // Average the loss
    } else if reduction == "sum" {
        // No further modification, already summed
    } else if reduction == "batchmean" {
        total_loss /= N; // For batchmean, divide by N
    } else {
        // If the reduction type is invalid, return 0.0 without printing
        return 0.0;
    }

    // Return the computed loss
    return total_loss;
}
