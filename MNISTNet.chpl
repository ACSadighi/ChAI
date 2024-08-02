use Tensor;

use Network;

use Reflection;

import TOML;
import Time;


class CNN : Module(?) {
    var conv1: owned Conv2D(eltType);
    var conv2: owned Conv2D(eltType);
    var dropout1: owned Dropout(eltType);
    var dropout2: owned Dropout(eltType);
    var flatten: owned Flatten(eltType);
    var fc1: owned Linear(eltType);
    var fc2: owned Linear(eltType);

    proc init(type eltType = real) {
        super.init(eltType);
        // (1,3,3) x 32
        this.conv1 = new Conv2D(eltType,channels=1,features=32,kernel=3,stride=1); // (1,X,X) -> (32,Y,Y)

        // (32,3,3) x 64
        this.conv2 = new Conv2D(eltType,channels=32,features=64,kernel=3,stride=1); // (32,Y,Y) -> (64,Z,Z)
        this.dropout1 = new Dropout(eltType,0.25);
        this.dropout2 = new Dropout(eltType,0.5);
        this.flatten = new Flatten(eltType);
        this.fc1 = new Linear(eltType,9216,128);
        this.fc2 = new Linear(eltType,128,10);

        init this;
        this.moduleName = "cnn";

        for (n,m) in moduleFields() {
            addModule(n,m);
        }
    }

    override proc forward(input: Tensor(eltType)): Tensor(eltType) {
        var st = new Time.stopwatch();
        //writeln("conv1");
        //st.start();
        var x = this.conv1(input);
        //st.stop();
        //writeln("conv1 ", //st.elapsed());
        //st.clear();
 

        //writeln("relu");
        //st.start();
        x = x.relu();
        //st.stop();
        //writeln("relu ", //st.elapsed());
        //st.clear();
        

        //writeln("conv2");
        //st.start();
        x = this.conv2(x);
        //st.stop();
        //writeln("conv2 ", //st.elapsed());
        //st.clear();

        //writeln("relu");
        //st.start();
        x = x.relu();
        //st.stop();
        //writeln("relu ", //st.elapsed());
        //st.clear();

        //writeln("maxpool");
        //st.start();
        x = x.maxPool(2);
        //st.stop();
        //writeln("maxpool ", //st.elapsed());
        //st.clear();

        //writeln("dropout1");
        //st.start();
        x = this.dropout1(x);
        //st.stop();
        //writeln("dropout ", //st.elapsed());
        //st.clear();

        //writeln("flatten");
        //st.start();
        x = x.flatten();
        //st.stop();
        //writeln("flatten ", //st.elapsed());
        //st.clear();

        //writeln("fc1");
        //st.start();
        x = this.fc1(x);
        //st.stop();
        //writeln("fc1 ", //st.elapsed());
        //st.clear();

        //writeln("relu");
        //st.start();
        x = x.relu();
        //st.stop();
        //writeln("relu ", //st.elapsed());
        //st.clear();

        //writeln("dropout2");
        //st.start();
        x = this.dropout2(x);
        //st.stop();
        //writeln("dropout2 ", //st.elapsed());
        //st.clear();
        
        //writeln("fc2");
        //st.start();
        x = this.fc2(x);
        //st.stop();
        //writeln("fc2 ", //st.elapsed());
        //st.clear();
        //st.start();

        //writeln("softmax");
        //st.start();
        var output = x.softmax();
        //st.stop();
        //writeln("softmax ", st.elapsed());
        //st.clear();

        return output;
    }
}
config const diag = false;


if diag {
    use GpuDiagnostics;

    startGpuDiagnostics();
    startVerboseGpu();
}


var cnn = new CNN(real);


for (n,m) in cnn.moduleFields() {
    writeln(n);
}


config const testImgSize = 28;

var img = Tensor.load("data/datasets/mnist/image_idx_0_7_7.chdata");// Tensor.arange(1,testImgSize,testImgSize);
// writeln(img);
// writeln(img.runtimeRank);

const modelPath = "data/models/mnist_cnn/";

cnn.loadPyTorchDump(modelPath);


// var output = cnn(img);

// writeln(output);

config const imageCount = 0;

var images = forall i in 0..<imageCount do Tensor.load("data/datasets/mnist/image_idx_" + i:string + ".chdata");
var preds: [images.domain] int;

config const numTimes = 1;
var conv1 = new Conv2D(real,channels=1,features=32,kernel=3,stride=1);
// 1000 : 0.081372
// 5000 : .4606

var conv2 = new Conv2D(real,channels=32,features=64,kernel=3,stride=1);


for i in 0..<numTimes {
    var st = new Time.stopwatch();
    st.start();
    forall (img,pred) in zip(images, preds) {
        // var x = conv1(img);
        // x = conv2(x);
        // var output = x;
        // pred = output.runtimeRank;
        var output: Tensor(real) = cnn(img);
        pred = output.argmax();
        // writeln((i, pred));
    }
    const tm = st.elapsed();
    st.stop();
    writeln("Time: ", tm, " seconds.");
}

config const printResults = false;
if printResults {
    for i in images.domain {
        writeln((i, preds[i]));
    }
}

var cnn2 = new Sequential(real,(
    new Conv2D(real,channels=1,features=32,kernel=3,stride=1)?
    ,new Conv2D(real,channels=32,features=64,kernel=3,stride=1)?
    ,new Dropout(real,0.25)?
    ,new Dropout(real,0.5)?
    ,new Flatten(real)?
    ,new Linear(real,9216,128)?
    ,new Linear(real,128,10)?)
        );
