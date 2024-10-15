use Tensor;

use Network;

use BlockDist;

import Time;

config const detach = true;

Tensor.detachMode(detach);

type dtype = real(32);

// Load an array of images. 
config const numImages = 1;

// Create distributed domain for images. 
const imagesD = blockDist.createDomain({0..<numImages});

// Load distributed array of images.
var images = forall i in imagesD do 
    Tensor.load("data/datasets/mnist/image_idx_" + i:string + ".chdata") : dtype;

// Create distributed domain for models.
const localeModelsD = blockDist.createDomain(Locales.domain);

// Load distributed array of models. 
var localeModels = forall li in localeModelsD do
    loadModel(specFile="scripts/models/cnn/specification.json",
              weightsFolder="scripts/models/cnn/",
              dtype=dtype);

// Create distributed array of output results.
var preds: [imagesD] int;


config const numTries = 1;

var totalTime: real;

for i in 0..<numTries {

    var st = new Time.stopwatch();
    st.start();

    // coforall loc in Locales {
    //     on loc {
    //         const myAD = A.domain.localIndices();
    //         forall i in myAD;
    //     }
    // }
    forall (image,pred) in zip(images,preds) {
        var model = localeModels[here.id].borrow();
        pred = model(image).argmax();
    }

    st.stop();
    const tm = st.elapsed();
    totalTime += tm;

    writeln("Trial ", i + 1, " of ", numTries," took ", tm, " seconds for ", numImages, " images on ", Locales.size, " nodes.");
}

const averageTime = totalTime / numTries;


config const printResults = false;
if printResults {
    for i in images.domain {
        writeln((i, preds[i]));
    }
}

writeln("The average inference time for batch of size ", numImages, " was ", averageTime, " seconds on ", Locales.size, " nodes.");
writeln("The total inference time for batch of size ", numImages, " was ", totalTime, " seconds on ", Locales.size, " nodes.");
