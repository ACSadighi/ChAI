use Random;
use losses;
use Math;

config const seed = 43;

proc main() {
    var A : [1..3, 1..2] real;
    fillRandom(A, seed);

    forall (i, j) in A.domain {
        A[i, j] = log(A[i, j]);
    }

    var target : [1..3] int = [1, 2, 1];
    var weight : [1..2] real = [2.0, 1.0];

    var input: [1..3] real = [0.5, 0.2, 0.3];
    var t: [1..3] real = [0.4, 0.3, 0.3];

    var nll = nll_loss(A, target, weight);
    var kl = kldiv_loss(input, t);

    writeln("NLLLoss: ", nll);
    writeln("KLDivLoss: ", kl);
}

main();