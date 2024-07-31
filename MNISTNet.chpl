use Tensor;

use Network;

use Reflection;

import TOML;


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

        this.conv1 = new Conv2D(eltType,channels=1,features=32,kernel=3,stride=1);
        this.conv2 = new Conv2D(eltType,channels=32,features=64,kernel=3,stride=1);
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

        // addModule("conv1", new Conv2D(eltType,channels=1,features=32,kernel=3,stride=1));
        // addModule("conv2", new Conv2D(eltType,channels=32,features=64,kernel=3,stride=1));
        // addModule("dropout1", new Dropout(eltType,0.25));
        // addModule("dropout2", new Dropout(eltType,0.5));
        // addModule("flatten", new Flatten(eltType));
        // addModule("fc1", new Linear(eltType,9216,128));
        // addModule("fc2", new Linear(eltType,128,10));
    }

    override proc forward(input: Tensor(eltType)): Tensor(eltType) {
                // writeln("conv1");
        var x = this["conv1"](input);

        // writeln("relu");
        x = x.relu();

        // writeln("conv2");
        x = this["conv2"](x);

        // writeln("relu");
        x = x.relu();

        // writeln("maxpool");
        x = x.maxPool(2);

        // writeln("dropout");
        x = this["dropout1"](x);

        // writeln("flatten");
        x = x.flatten();

        // writeln("fc1");
        x = this["fc1"](x);

        // writeln("relu");
        x = x.relu();

        // writeln("dropout");
        x = this["dropout1"](x);

        // writeln("fc2");
        x = this["fc2"](x);
        // // writeln("conv1");
        // var x = mod("conv1")(input);

        // // writeln("relu");
        // x = x.relu();

        // // writeln("conv2");
        // x = mod("conv2")(x);

        // // writeln("relu");
        // x = x.relu();

        // // writeln("maxpool");
        // x = x.maxPool(2);

        // // writeln("dropout");
        // x = mod("dropout1")(x);

        // // writeln("flatten");
        // x = x.flatten();

        // // writeln("fc1");
        // x = mod("fc1")(x);

        // // writeln("relu");
        // x = x.relu();

        // // writeln("dropout");
        // x = mod("dropout1")(x);

        // // writeln("fc2");
        // x = mod("fc2")(x);

        // writeln("softmax");
        var output = x.softmax();
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


var output = cnn(img);

// writeln(output);

config const imageCount = 0;

var images = for i in 0..<imageCount do Tensor.load("data/datasets/mnist_dump2/image_idx_" + i:string + ".chdata");
var preds: [images.domain] int;

writeln(images[1]);

for i in images.domain {
    var output = cnn(images[i]);
    var pred = output.argmax();
    preds[i] = pred;
    writeln((i, pred));
}

config const printResults = true;
if printResults {
    for i in images.domain {
        writeln((i, preds[i]));
    }
}

writeln(cnn.conv1.attributes());
writeln(cnn.conv1.attributes().prettyPrint());

var cnn2 = new Sequential(real,(
    new Conv2D(real,channels=1,features=32,kernel=3,stride=1)?
    ,new Conv2D(real,channels=32,features=64,kernel=3,stride=1)?
    ,new Dropout(real,0.25)?
    ,new Dropout(real,0.5)?
    ,new Flatten(real)?
    ,new Linear(real,9216,128)?
    ,new Linear(real,128,10)?)
        );



// import IO;
// import JSON;

// const tomlFile = IO.open("data/models/mnist_cnn/summary.toml", IO.ioMode.r);
// const toml = TOML.parseToml(tomlFile);
// writeln(toml);

// writeln(toml["cnn"]);

// var jsw = IO.stdout.withSerializer(JSON.jsonSerializer);
// toml.writeJSON(jsw);
// toml.writeTOML(jsw);
// toml.writeJSON(IO.stdout);



// var model = new Sequential(real,new Conv2D(real,channels=1,features=32,kernel=3,stride=1));

// param nf = getNumFields(cnn.type);
// for param i in 0..<nf {
//     param name = getFieldName(cnn.type,i);
//     writeln(name,getField(cnn,name).type:string);
// }
// for mn in model.moduleNames() {
//     writeln(mn);
// }
// writeln(model(img));


