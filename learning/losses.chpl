use Math;

class NLLLoss {
  var weight: [] real; // Class weights
  var reduction: string; // Reduction type: 'none', 'mean', or 'sum'
  var ignoreIndex: int; // Index to ignore during computation

  // Constructor
  proc init(weight: [] real = [], reduction: string = "mean", ignoreIndex: int = -100) {
    this.weight = weight;
    this.reduction = reduction;
    this.ignoreIndex = ignoreIndex;
  }

  // Forward pass
  proc forward(input: [][?N] real, target: [?T] int): real {
    var batchSize = target.size;
    var loss: [1..batchSize] real;

    // Compute the loss for each target
    for i in 1..batchSize {
      var targetClass = target[i];
      if targetClass == this.ignoreIndex {
        loss[i] = 0.0; // Ignore this index
      } else {
        var weightFactor = if this.weight.size > 0 then this.weight[targetClass] else 1.0;
        loss[i] = -weightFactor * input[i, targetClass];
      }
    }

    // Apply reduction
    if this.reduction == "mean" {
      var totalWeight = sum(weight for weight in this.weight if weight > 0);
      return totalWeight > 0 ? sum(loss) / totalWeight : sum(loss) / batchSize;
    } else if this.reduction == "sum" {
      return sum(loss);
    } else {
      return loss; // No reduction
    }
  }
}


class KLDivLoss {
  var reduction: string; // Reduction type: 'none', 'mean', 'sum', or 'batchmean'
  var logTarget: bool;   // Whether the target is in log-space

  // Constructor
  proc init(reduction: string = "mean", logTarget: bool = false) {
    this.reduction = reduction;
    this.logTarget = logTarget;
  }

  // Forward pass
  proc forward(input: [][?N] real, target: [][?M] real): real {
    if input.size != target.size {
      halt("Input and target must have the same shape");
    }

    var loss_pointwise: [1..input.size] real;

    // Compute the pointwise KL divergence
    if this.logTarget {
      // log_target = True
      for i in 1..input.size {
        loss_pointwise[i] = exp(target[i]) * (target[i] - input[i]);
      }
    } else {
      // log_target = False
      for i in 1..input.size {
        loss_pointwise[i] = target[i] * (log(target[i]) - input[i]);
      }
    }

    // Apply reduction
    if this.reduction == "mean" {
      return mean(loss_pointwise);
    } else if this.reduction == "sum" {
      return sum(loss_pointwise);
    } else if this.reduction == "batchmean" {
      return sum(loss_pointwise) / input.size;
    } else {
      // reduction = 'none'
      return loss_pointwise;
    }
  }
}
