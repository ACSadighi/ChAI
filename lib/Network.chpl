// use ndarray;
// use remote;
// use autograd;
// import autograd as ag;
// import StaticTensor as tn;
// use StaticTensor;

// use DynamicTensor

prototype module Network {
use Tensor;

use Map; // only map;
use List;
use OrderedDict;

use Reflection;



proc helpFindModuleByName(arg, x: string) : borrowed Module(?)? {
  param myFields = getNumFields(arg.type);
  for param i in 0..<myFields {
    if !isType(getField(arg, i)) &&
      isSubtype(getField(arg, i).type, Module) &&
        getFieldName(arg.type, i) == x {
          return getField(arg, i).borrow();
        }
  }
  halt("Could not find module with name: ", x);
  return nil;
}

proc helpFindParamDataByName(arg, x: string) ref : Tensor(?) {
  param myFields = getNumFields(arg.type);
  for param i in 0..<myFields {
    if !isType(getField(arg, i)) &&
      isSubtype(getField(arg, i).type, Tensor(?)) &&
        getFieldName(arg.type, i) == x {
          return getField(arg, i).borrow();
        }
  }
  return new Tensor(?);
}

record moduleChildren {
    type eltType = real;
    var childDict: map(string,borrowed Module(eltType));
    var order: list(string);

    proc init(type eltType = real) {
        this.eltType = eltType;
        this.childDict = new map(string,borrowed Module(eltType),initialCapacity=1);
        this.order = new list(string);
    }

    iter ref these(): borrowed Module(eltType) do
        for k in 0..<order.size do
            yield childDict[order(k)];

    iter ref items(): (string,borrowed Module(eltType)) do
        for n in 0..<order.size do
            yield (order(n),childDict[order(n)]);

    iter ref itemsPar(): (string,borrowed Module(eltType)) do
        foreach n in 0..<order.size do
            yield (order(n),childDict[order(n)]);

    proc ref add(name: string,m: borrowed Module(eltType)) {
        order.pushBack(name);
        childDict.addOrReplace(name,m);
    }

    proc ref ith(i: int): borrowed Module(eltType) do 
        return childDict(order(i));

}




proc (class).this(fieldName: string): borrowed Module(?) where isSubtype(this.type,Module(?)) {
    return helpFindModuleByName(this,fieldName)!;
}

proc (class).this(fieldName: string) ref : Tensor(?) where isSubtype(this.type,Parameter(?)) {
    return helpFindParamDataByName(this,fieldName);
}


iter (class).moduleFieldNames(): string where isSubtype(this.type,Module(?)) {
    param myFields = getNumFields(this.type);
    for param i in 0..<myFields {
        param fieldName = getFieldName(this.type, i);
        if !isType(getField(this, i)) && isSubtype(getField(this, i).type, Module(?)) {
            yield fieldName;
        }
    }
}

iter (class).moduleFields(): (string,borrowed Module(?)) where isSubtype(this.type,Module(?)) {
    for mn in this.moduleFieldNames() {
        yield (mn,this[mn]);
    }
}

proc (class).registerModules() where isSubtype(this.type,Module(?)) {
    for (n,m) in this.moduleFields() {
        addModule(n,m);
    }
}


proc (class).postinit() where isSubtype(this.type,Module(?)) {
    for (n,m) in this.moduleFields() {
        addModule(n,m);
    }
}




record moduleAttributes : serializable {
    var layerType: string;
    var moduleName: string;
    var attributes: dict(string,string);
    forwarding attributes only this;

    proc init(layerType: string,moduleName: string,in attrs: map(string,string,?)) {
        this.layerType = layerType;
        this.moduleName = moduleName;
        this.attributes = new dict(attrs);
    }

    proc init(layerType: string,moduleName: string,in attrs: dict(string,?)) {
        this.layerType = layerType;
        this.moduleName = moduleName;
        this.attributes = new dict(string,string);
        init this;
        for (k,v) in attrs do 
            attributes.insert(k,v : string + "\n\t");
    }

    proc init(layerType: string,moduleName: string,in attrs: map(string,?valType,?)) where valType != string {
        this.layerType = layerType;
        this.moduleName = moduleName;
        this.attributes = new dict(string,string);
        init this;
        for k in attrs.keys() do
            attributes.insert(k,attrs[k] : string + "\n\t");
    }

    proc init(layerType: string,moduleName: string,in attrs: map(string,?valType,?),order: list(string)) {
        this.layerType = layerType;
        this.moduleName = moduleName;
        this.attributes = new dict(string,string);
        init this;
        for i in 0..<order.size {
            const k = order(i);
            attributes.insert(k,attrs[k] : string + "\n\t");
        }
    }

    pragma "last resort"
    proc init(layerType: string,moduleName: string, attrs...?n) where attrs(0)(0).type == string {
        this.layerType = layerType;
        this.moduleName = moduleName;
        this.attributes = new dict(string,string);
        init this;
        for param i in 0..<n {
            attributes.insert(attrs(i)(0),attrs(i)(1) : string);
        }
    }

    proc init(layerType: string,moduleName: string) {
        this.layerType = layerType;
        this.moduleName = moduleName;
        this.attributes = new dict(string,string);
    }

    proc getInt(name: string): int do
        return attributes[name] : int;

    proc prettyPrint(): string {
        var s: string = layerType + "(";
        const size = attributes.size;
        var idx = 0;
        for (k,v) in attributes {
            s += k + " = " + v;
            if idx < size - 1 then
                s += ", ";
            idx += 1;
        }
        s += ")";
        return s;
    }

    proc prettyPrintSpec(): string do
        return moduleName + " : " + prettyPrint();

    operator :(ma: moduleAttributes, type T: string) {
        return ma.prettyPrint();
        // return ma.prettyPrintSpec();
    }
}

class ModuleSpecification : serializable {
    var layerType: string;
    var attributes: map(string,string);
    var subModules: map(string,owned ModuleSpecification?);
    var subModuleOrder: list(string);
}

@chpldoc.nodoc
const empty_locales: [1..0] locale;

@chpldoc.nodoc
record optional {
    type t;
    var x: t;
    var isSome: bool;
    proc type some(x: ?t): optional(t) {
        return new optional(t,x,true);
    }
    proc type empty(type t): optional(t) {
        var x: t;
        return new optional(t,x,false);
    }
}

proc moduleFromSpec(
    ms_: borrowed ModuleSpecification?,
    type dtype = real(32),
    targetLocales: [] locale = empty_locales,
    inputShape: optional(?) = optional.empty(1*int)
): owned Module(dtype) {
    var ms = ms_!;
    select ms.layerType {
        when "Conv2d" {
            return new Conv2D(dtype,new moduleAttributes(ms.layerType,"unknown",ms.attributes));
        }
        when "Linear" {
            var ma = new moduleAttributes("Linear","unknown",ms.attributes);
            return new Linear(dtype,ma.getInt("in_features"),ma.getInt("out_features"));
        }
        when "Dropout" {
            var ma = new moduleAttributes("Dropout","unknown",ms.attributes);
            return new Dropout(dtype,ma["p"] : dtype);
        }
        when "Flatten" {
            return new Flatten(dtype);
        }
        when "ReLU" {
            return new ReLU(dtype);
        }
        when "MaxPool2d" {
            var ma = new moduleAttributes("MaxPool","unknown",ms.attributes);
            return new MaxPool(dtype,ma.getInt("kernel_size"),ma.getInt("stride"));
        }
        when "AdaptiveAvgPool2d" {
            var ma = new moduleAttributes("AdaptiveAvgPool2D","unknown",ms.attributes);
            return new AdaptiveAvgPool2D(dtype,ma.getInt("output_size"));
        }
        when "LogSoftmax" {
            return new Softmax(dtype);
        }
        otherwise {
            var sms: dict(string,shared Module(dtype)) = new dict(string,shared Module(dtype));
            if targetLocales.size > 0 {
                import this.SubModDistribution.{partitionModulesAcrossLocales, interModuleComms, memorySize, printPartitioningInfo};

                // split the modules into groups across locales, keeping the amount of data roughly balanced
                // and minimizing the amount of communication between locales
                const subModSizes = [k in ms.subModuleOrder] memorySize(ms.subModules[k]!),
                      subModComms = interModuleComms(ms, inputShape),
                      locPartition = partitionModulesAcrossLocales(subModSizes, subModComms, targetLocales.size, 0.75, 0.25),
                      layerTypes = [k in ms.subModuleOrder] ms.subModules[k]!.layerType;

                printPartitioningInfo(layerTypes, subModSizes, subModComms, locPartition, targetLocales);

                for i in 0..<targetLocales.size {
                    for k in locPartition[i]..<locPartition[i+1] {
                        const sma = ms.subModules[ms.subModuleOrder[k]];
                        on targetLocales[targetLocales.domain.orderToIndex(i)] {
                            const sm: shared Module(dtype) = shared.adopt(moduleFromSpec(sma,dtype=dtype));
                            sms.insert(ms.subModuleOrder[k],sm);
                        }
                    }
                }
            } else {
                for k in ms.subModuleOrder {
                    const sma = ms.subModules[k];
                    const sm: shared Module(dtype) = shared.adopt(moduleFromSpec(sma,dtype=dtype));
                    sms.insert(k,sm);
                }
            }

            return new Sequential(dtype, sms, moduleName=ms.layerType);
        }
    }
    halt("This should not happen");
}

proc modelFromSpecFile(path: string, type dtype=real(32), targetLocales: [] locale = empty_locales, inputShape: optional(?), debug = false) : owned Module(dtype) {
    import IO;
    import JSON;
    var fl = IO.open(path, IO.ioMode.r);
    var reader = fl.reader(deserializer=new JSON.jsonDeserializer());
    var ms = reader.read(owned ModuleSpecification);
    fl.close();
    return moduleFromSpec(ms,dtype,targetLocales,inputShape);
}

proc loadModel(specFile: string, weightsFolder: string, type dtype = real(32),param debug = false): owned Module(dtype) {
    var model: owned Module(dtype) = modelFromSpecFile(specFile, dtype, empty_locales, optional.empty(1*int));

    model.loadPyTorchDump(weightsFolder,dtype = dtype, debug = debug);
    return model;
}

proc loadModelAcrossLocales(specFile: string, weightsFolder: string, targetLocales: [] locale, inputShape: ?N*int, type dtype = real(32)): owned Module(dtype) {
    var model: owned Module(dtype) = modelFromSpecFile(specFile, dtype, targetLocales, optional.some(inputShape));

    model.loadPyTorchDump(weightsFolder);
    return model;
}

// helper procedures for intelligently partitioning modules across locales
private prototype module SubModDistribution {
    use List;
    import super.{ModuleSpecification, optional};

    // parameters for genetic optimization used below
    private const generationSize = 20,
                poolSize = 2,
                numGenerations = 20;

    /*
        Partition 'n' sub-modules across 'N' locales, minimizing both the variance in
        the total amount of sub-module data on each locale, and the total amount of
        communication between locales. In other words, partitions should be chosen such
        that the amount of data stored on each locale is roughly even, and the amount
        of data communicated between locales (by sub-modules that are sequential in the
        network, but located on different locales) is minimized.

        The cost function is a weighted sum of the normalized variance in the total
        amount of data on each locale, and the normalized total amount of communication
        between locales. The weights are given by 'alpha' and 'beta', respectively.

        Returns an array 'p' of 'N+1' integers, where 'p[i]..<p[i+1]' represents the range
        of sub-module indices assigned to the ith locale. The first element is always 0,
        and the last element is always 'n' (where 'n' is the number of sub-modules).

        This procedure uses a simple genetic algorithm to search for an optimal partitioning.
    */
    proc partitionModulesAcrossLocales(
        const ref sizes: [?d] int,
        const ref comms: [d] int,
        N: int,
        alpha: real,
        beta: real
    ): [] int {
        use Random;

        const n = d.size; // number of sub-modules

        // mean data-size-per-locale (equivalent to mean-data-size multiplied by (n/N))
        const meanSize: real = (+ reduce sizes):real / N;

        // variance of per-locale data-size for a given partitioning
        proc sizeVariance(const ref p: [] int): real {
            var ssum = 0.0;
            for i in 0..<N do
                ssum += ((+ reduce sizes[p[i]..<p[i+1]]) - meanSize) ** 2;
            return sqrt(ssum / (N - 1));
        }

        // total amount of comm between locales for a given partitioning
        proc totalComm(const ref p: [] int): int {
            return + reduce for i in 1..<N do comms[p[i]];
        }

        // overall cost function
        const sizeSum = + reduce sizes,
            commSum = + reduce comms;
        proc cost(const ref p): real {
            return alpha * sizeVariance(p) / sizeSum +
                beta * totalComm(p) / commSum;
        }

        // yield 'popSize' random mutations from the given genetic pool
        iter randomMutations(pool /* : [] [] int */, popSize: int, ref rs: randomStream(int)): [] int {
            const deltaP = N / 2; // max change in partitioning
            var delta: [pool[0].domain] int,
                count = 0;
            while count < (popSize - pool.size) {
                // generate a random mutation of a random sample from the pool
                rs.fill(delta, -deltaP, deltaP + 2);
                for d in delta do if d > deltaP then d = 0; // simple way of making 0 more likely
                var pm: [pool[0].domain] int = rs.choose(pool) + delta;
                pm[0] = 0;
                pm[N] = n;

                // only yield valid partitions
                if && reduce for i in 0..<N do pm[i] < pm[i+1] {
                    yield pm;
                    count += 1;
                }
            }

            // yield the original pool
            for p in pool do yield p;
        }

        // initial guess for partitioning: give each locale a roughly equal number of sub-modules
        proc goodGuess(): [] int {
            var p: [0..N] int;
            const q = n / N,
                r = n % N;
            p[0] = 0;
            p[N] = n;
            for i in 1..<N do
                p[i] = p[i-1] + q + if i <= r then 1 else 0;
            return p;
        }

        var pmin = goodGuess(),
            rs = new randomStream(int, seed=314159),
            pool = for i in 0..poolSize do pmin;

        for 1..numGenerations {
            const population = randomMutations(pool, generationSize, rs),
                costs = for p in population do cost(p),
                (_, minIdx) = minloc reduce zip(costs, costs.domain);

            pmin = population[minIdx];

            // new pool with the best and a few random mutations
            pool[0] = pmin;
            pool[1..] = rs.sample(population, poolSize);
        }

        return pmin;
    }

    record commSize {
        var sizes: list(int);

        proc init() {
            this.sizes = new list(int);
        }

        proc init(in sizes: [] int) {
            this.sizes = new list(sizes);
        }

        proc init(x: int ...?N) {
            this.sizes = new list(int);
            init this;
            for i in 0..<N do
                sizes.pushBack(x[i]);
        }

        proc size: int {
            var total = 1;
            for s in sizes do total *= s;
            return total;
        }

        proc this(x: int): int {
            return sizes[x];
        }
    }

    /*
        using a unit-sized input tensor, compute the size of the output tensor for each sub-module
    */
    proc interModuleComms(ms: borrowed ModuleSpecification, inputShape: optional(?t)): [] int throws {
        const n = ms.subModuleOrder.size;
        var comms: [-1..<n] commSize;

        // create a 'commSize' representing the size of the network's input tensor
        const ir = inputRank(ms.subModules[ms.subModuleOrder[0]]!);
        var inputShape_ = [i in 0..<ir] 1;
        if inputShape.isSome && isHomogeneousTuple(t) {
            if inputShape.x.size != ir then
                throw new Error("rank of provided 'inputShape' does not match module's input rank");
            inputShape_ = [i in 0..<ir] inputShape.x[i];
        }
        comms[-1] = new commSize(inputShape_);

        // compute the output size of each module
        for i in 0..<n do comms[i] = outputSize(ms.subModules[ms.subModuleOrder[i]]!, comms[i-1]);

        const moduleComms = [i in 0..<n] comms[i].size;
        return moduleComms;
    }

    // Get the number of elements stored in the module's tensors.
    // Used to evenly partition modules across locales
    // note: when a new module with a non-trivial memory footprint is added,
    //       a corresponding when-clause should be added here
    proc memorySize(ms: borrowed ModuleSpecification): int {
        select ms.layerType {
            when "Conv2d" {
                const ic = ms.attributes["in_channels"]: int,
                      oc = ms.attributes["out_channels"]: int,
                      ks = ms.attributes["kernel_size"]: int;
                //            weights    + bias
                return ic * oc * ks * ks + oc;
            }
            when "Linear" {
                const ifs = ms.attributes["in_features"]: int,
                      ofs = ms.attributes["out_features"]: int;
                //       weights + bias
                return ifs * ofs + ofs;
            }
            otherwise {
                return 0;
            }
        }
    }

    // Get the number of elements in the output tensor of the module
    // note: when a new module whose output shape doesn't match its input shape is added,
    //       a corresponding when-clause should be added here
    proc outputSize(ms: borrowed ModuleSpecification, inputSize: commSize): commSize {
        select ms.layerType {
            when "Conv2d" {
                const oc = ms.attributes["out_channels"]: int,
                      ks = ms.attributes["kernel_size"]: int,
                      s = ms.attributes["stride"]: int;

                const outHeight = ((inputSize[1] - ks) / s) + 1,
                      outWidth = ((inputSize[2] - ks) / s) + 1;

                return new commSize(oc, outHeight, outWidth);
            }
            when "Linear" {
                return new commSize(ms.attributes["out_features"]: int);
            }
            when "MaxPool" {
                const ps = ms.attributes["pool_size"]: int,
                      s = ms.attributes["stride"]: int;

                const channels = inputSize[0],
                      outHeight = ((inputSize[1] - ps) / s) + 1,
                      outWidth = ((inputSize[2] - ps) / s) + 1;

                return new commSize(channels, outHeight, outWidth);
            }
            when "AdaptiveAvgPool2D" {
                const os = ms.attributes["output_size"]: int;
                return new commSize(inputSize[0], os, os);
            }
            when "Flatten" do return new commSize(inputSize.size);
            otherwise do return inputSize;
        }
    }

    // rank of the tensor that each layer type takes as an input
    // used for computing the amount of data communicated between layers
    proc inputRank(ms: borrowed ModuleSpecification): int {
        select ms.layerType {
            when "Conv2d" do return 3;
            when "Linear" do return 1; // (not true, but works for communication estimation)
            when "MaxPool" do return 3;
            when "AdaptiveAvgPool2D" do return 3;
            when "Flatten" do return 1;
            otherwise do return 1;
        }
    }

    proc printPartitioningInfo(names: [?d] string, sizes: [d] int, comms: [d] int, partition: [?dp] int, locales: [?dl] locale) {
        proc nChars(x: int): int(8) {
            use Math;
            return ceil(log10(x:real)):int(8);
        }

        var maxNChars = 0;
        for n in names do maxNChars = max(maxNChars, n.size);
        for s in sizes do maxNChars = max(maxNChars, nChars(s));
        for c in comms do maxNChars = max(maxNChars, nChars(c));

        const charsPerLoc = [i in dl] (maxNChars+1) * (partition[i+1] - partition[i]);

        proc printInfoLine(const ref a, name: string, fmt: string) {
            write(name, " |");
            for p in 0..<dp.size-1 {
                for i in partition[p]..<partition[p+1] do
                    writef("%*" + fmt + " ", maxNChars, a[i]); // "
                write("|");
            }
            writeln();
        }

        writeln("sub-module distribution");
        writeln("-----------------------");
        write("      |"); for (loc, i) in zip(locales, dl) do
            writef(" %<*s|", charsPerLoc[i]-1, "LOCALE"+loc.id:string); writeln();

        printInfoLine(names, "layer", "s");
        printInfoLine(sizes, "size ", "i");
        printInfoLine(comms, "comms", "i");
        writeln("-----------------------");
    }
}

var moduleInstances = 0;

class Module {
    type eltType;
    var subModules: moduleChildren(eltType);
    var moduleId: int;
    var moduleName: string;
    var ownedModules: list(shared Module(eltType));

    proc init(type eltType = real) {
        this.eltType = eltType;
        this.subModules = new moduleChildren(eltType);
        this.moduleId = moduleInstances;
        this.moduleName = "module[" + moduleInstances:string + "]";
        this.ownedModules = new list(shared Module(eltType));
        moduleInstances += 1;
    }

    proc init(type eltType = real,ma: moduleAttributes) do
        this.init(eltType);

    proc setup() { }

    proc this(input: Tensor(eltType)): Tensor(eltType) do
        return this.forward(input);

    proc getSubModuleName(name: string): string do
        return moduleName + "." + name;

    proc addModule(name: string, m: borrowed Module(eltType)) {
        const modName = getSubModuleName(name);
        m.moduleName = modName;
        m.setup();
        subModules.add(modName,m);
    }

    proc addModule(name: string, m: shared Module(eltType)) {
        ownedModules.pushBack(m);
        addModule(name,m.borrow());
    }
    pragma "last resort"
    proc addModule(name: string, in m: owned Module(eltType)) {
        var sm: shared Module(eltType) = shared.adopt(m);
        addModule(name,sm);
    }

    proc addParameter(name: string, data: Tensor(eltType)) {
        const modName = getSubModuleName(name);
        var p = new owned Parameter(data);
        p.moduleName = modName;
        p.setup();
        subModules.add(modName,p);
    }

    proc forward(input: Tensor(eltType)): Tensor(eltType) do
        halt("Unimplemented.");

    proc par(paramName: string) ref : Tensor(eltType) {
        return (subModules.childDict[getSubModuleName(paramName)] : borrowed Parameter(eltType)).data;
    }

    proc mod(modName: string): borrowed Module(eltType) {
        return subModules.childDict[getSubModuleName(modName)];
    }

    iter parameters(): borrowed Parameter(eltType) {
        for m in modules() {
            if var p = m : borrowed Parameter(eltType)? then
                yield p!;
        }
    }

    iter moduleNames(): string {
        for m in modules() {
            yield m.moduleName;
        }
    }

    iter parameterNames(): string {
        for p in parameters() {
            yield p.moduleName;
        }
    }

    iter modules(): borrowed Module(eltType) {
        for m in subModules {
            yield m;
            for m_ in m.modules() {
                yield m_;
            }
        }
    }

    iter namedModules(): (string,borrowed Module(eltType)) {
        for (n,m) in subModules.items() {
            yield (n,m);
            for (n_,m_) in m.namedModules() {
                yield (n_,m_);
            }
        }
    }
    iter namedModules(param tag: iterKind): (string,borrowed Module(eltType)) where tag == iterKind.standalone {
        foreach (n,m) in subModules.itemsPar() {
            yield (n,m);
            foreach (n_,m_) in m.namedModules() {
                yield (n_,m_);
            }
        }
    }

    // iter modules(): borrowed Module(eltType) {
    //     for (n,m) in this.moduleFields()
    // }


    proc loadSingleParameter(parameter: Parameter(?), mod: Module(?), modelPath: string, prefix: string = "", indent: string, param debug, type dtype = real(32)) {
        const paramName = parameter.moduleName;
        const name = prefix + paramName[(mod.moduleName.size + 1)..];
        const paramPath = modelPath + name + ".chdata";
        if debug then
            writeln(indent,"Loading ", name, " from ", paramPath);
        parameter.data = Tensor.load(paramPath, debug=debug, dtype=dtype) : eltType;
    }

    // Define the loadParameters function
    proc loadParameters(mod, modelPath: string, prefix: string = "", indent: string, param debug, type dtype = real(32)) {
        for (_, m) in mod.subModules.items() {
            const modName = m.moduleName;
            if var p = m : borrowed Parameter(eltType)? {
                loadSingleParameter(p, mod, modelPath, prefix, indent+"\t", debug, dtype);
            }
            else if var sm = m : borrowed Sequential(eltType)? {
                // Split the modName by the first period
                // And add the second part to the prefix
                const msplit = modName.split(".", maxsplit=2);
                loadParameters(m, modelPath, prefix + msplit[1] + ".", indent+"\t", debug, dtype);
            } else if var sm = m : borrowed Module(eltType)? {
                // Split the modName by the first period
                // And add the second part to the prefix
                const msplit = modName.split(".", maxsplit=1);
                loadParameters(m, modelPath, prefix + msplit[1] + ".", indent+"\t", debug, dtype);
            } else {
                halt("Unknown module type: ", m);
            }
        }
    }

    // Modify the loadPyTorchDump method to call loadParameters
    proc loadPyTorchDump(modelPath: string, param debug = false, type dtype = real(32)) {
        loadParameters(this, modelPath, prefix="", indent="", debug, dtype);
    }


    proc attributes(): moduleAttributes {
        var ms = new map(string,moduleAttributes);
        for (n,m) in subModules.items() {
            ms.addOrReplace(n,m.attributes());
        }
        return new moduleAttributes(
            "Module",
            moduleName,
            ms,
            subModules.order
        );
    }

    proc signature: string do
        return attributes().prettyPrint();
}

class Parameter : Module(?) {
    var data: Tensor(eltType);

    proc init(data: Tensor(?eltType)) {
        super.init(eltType);
        this.data = data;
    }

    override proc attributes(): moduleAttributes do
        return new moduleAttributes(
            "Parameter",
            moduleName,
            ("data", "<tensor>"));
}

class Sequential : Module(?) {
    var mds: list(shared Module(eltType));

    proc init(type eltType = real, ms: dict(string,shared Module(eltType)), moduleName: string = "sequential") {
        super.init(eltType);
        this.mds = new list(shared Module(eltType));
        init this;

        this.moduleName = moduleName;
        for (name,m) in ms {
            addModule(name,m.borrow());
            mds.pushBack(m);
        }
    }

    proc init(type eltType = real, in ms, moduleName: string = "sequential") {
        super.init(eltType);
        this.mds = new list(shared Module(eltType));
        init this;
        this.moduleName = moduleName;
        for param i in 0..<ms.size {
            var m : shared Module(eltType) = shared.adopt(owned.release(ms[i])!);
            addModule(i: string,m.borrow());
            mds.pushBack(m);
        }
    }

    proc init(type eltType = real, order: list(string), in ms: map(string,owned Module(eltType)?)) {
        super.init(eltType);
        this.mds = new list(shared Module(eltType));
        init this;
        this.moduleName = "sequential";
        for (i,k) in zip(0..<ms.size,ms.keys()) {
            // var m : owned Module(eltType) = owned.adopt(owned.release(ms[order[i]])!);
            var m : shared Module(eltType) = shared.adopt(owned.release(ms[k])!);
            const j = mds.pushBack(m);
            var b = mds[j].borrow();
            addModule(order[i],b);
            compilerWarning("Iain you need to fix this after the demo.");
        }
    }

    proc init(in ms: (owned Module(real)?)...?rank) do
        this.init(real, ms);

    proc init(type eltType, moduleName: string = "sequential") {
        super.init(eltType);
        this.mds = new list(shared Module(eltType));
        init this;

        this.moduleName = moduleName;
    }

    proc addModule(name: string, m: shared Module(eltType)) {
        mds.pushBack(m);
        addModule(name,m.borrow());
    }


    override proc forward(input: Tensor(eltType)): Tensor(eltType) {
        // for m in mds {
        //     writeln((m.moduleName, m.signature));
        // }
        // return input;
        // for (n,m) in this.namedModules() {
        //     writeln((n,m.moduleName,))
        // }
        if mds.size < 1 then
            halt("Sequential must have submodules! moduleName: ", moduleName);
        var x = mds[0](input);
        for i in 1..<mds.size {
            x = mds[i](x);
        }
        return x;
    }

    override proc attributes(): moduleAttributes {
        var ms = new dict(string,moduleAttributes);
        for (n,m) in subModules.items() {
            ms.insert(n,m.attributes());
        }
        return new moduleAttributes(
            "Sequential",
            moduleName,
            ms
        );
    }
}

class Linear : Module(?) {
    var m,n: int;
    var weight: owned Parameter(eltType);
    var bias: owned Parameter(eltType);

    proc init(type eltType, m: int, n: int) {
        super.init(eltType);
        this.m = m;
        this.n = n;
        this.weight = new Parameter(Tensor.arange(n,m) : eltType);
        this.bias = new Parameter(Tensor.zeros(m) : eltType);
        init this;

    }

    override proc setup() {
        addModule("weight",weight);
        addModule("bias",bias);
    }

    proc init(m: int, n: int) {
        this.init(real,m,n);
    }

    override proc forward(input: Tensor(eltType)): Tensor(eltType) do
        return Tensor.matvecmulFast(weight.data,input) + bias.data;

    override proc attributes(): moduleAttributes {
        return new moduleAttributes(
            "Linear",
            moduleName,
            ("inFeatures", m),
            ("outFeatures", n));
    }
}

class Conv2D : Module(?) {
    var kernelShape: 4*int;
    var stride: int;
    var padding: int;
    var kernel: owned Parameter(eltType);
    var bias: owned Parameter(eltType)?;

    proc init(type eltType = real,channels: int, features: int, kernel: int, stride: int = 1, padding: int = 0, bias: bool = true) {
        super.init(eltType);
        this.kernelShape = (features,channels,kernel,kernel);
        this.stride = stride;
        this.padding = padding;
        this.kernel = new Parameter(Tensor.arange(features,channels,kernel,kernel) : eltType);
        if bias then
            this.bias = new Parameter(Tensor.arange(features) : eltType);
        else
            this.bias = nil;
        init this;
    }

    proc init(type eltType = real,ma: moduleAttributes) {
        this.init(eltType,
                  ma.getInt("in_channels"),
                  ma.getInt("out_channels"),
                  ma.getInt("kernel_size"),
                  ma.getInt("stride"),
                  ma.getInt("padding"));
    }

    // proc init(reader,ref deserializer: jsonDeserializer) {
    //     var des = deserializer.startClass(reader,"Conv2D");
    //     const attributes = new moduleAttributes(des.readField("attributes",map(string,string)));

    //     // const channels = des.readField("inChannels",int);
    //     // const features = des.readField("outChannels",int);
    //     // const kernel = des.readField("kernel",int);
    //     // const stride = des.readField("stride",int);

    // }

    override proc setup() {
        // const (features,channels,kernel,_) = kernelShape;
        // var ker = Tensor.arange(features,channels,kernel,kernel);
        // var bias = Tensor.arange(features);

        addModule("weight",kernel);
        if bias != nil then
            addModule("bias",bias!);
    }

    proc init(channels: int, features: int, kernel: int, stride: int = 1, padding: int = 0) {
        this.init(real,channels,features,kernel,stride,padding);
    }

    override proc forward(input: Tensor(eltType)): Tensor(eltType) {
        var weights = this.kernel.data;
        var bias = if this.bias != nil then this.bias!.data else Tensor.zeros(this.kernelShape[0]):eltType;
        return Tensor.convolve(input,weights,bias,stride,padding);
    }

    override proc attributes(): moduleAttributes {
        const (features,channels,kernel,_) = kernelShape;
        return new moduleAttributes(
            "Conv2D",
            moduleName,
            ("inChannels", channels),
            ("outChannels", features),
            ("kernelSize", kernel),
            ("stride",stride),
            ("padding",padding));
    }
}

class MaxPool : Module(?) {
    var poolSize: int;
    var stride: int;
    var padding: int;
    var dilation: int;

    proc init(type eltType = real, poolSize: int, stride: int = -1, padding: int = 0, dilation: int = 1) {
        super.init(eltType);
        this.poolSize = poolSize;
        if stride == -1 then
          this.stride = poolSize;
        else
          this.stride = stride;
        this.padding = padding;
        this.dilation = dilation;
    }

    proc init(poolSize: int, stride: int = -1, padding: int = 0, dilation: int = 1) do
        this.init(real,poolSize, stride, padding, dilation);

    override proc forward(input: Tensor(eltType)): Tensor(eltType) {
        return input.maxPool(poolSize, stride, padding, dilation);
    }

    override proc attributes(): moduleAttributes {
        return new moduleAttributes(
            "MaxPool",
            moduleName,
            ("poolSize", poolSize),
            ("stride", stride),
            ("padding", padding),
            ("dilation", dilation));
    }
}

class AdaptiveAvgPool2D : Module(?) {
  // only handles square pooling
  var outputSize: int;

  proc init(type eltType = real, outputSize: int) {
        super.init(eltType);
        this.outputSize = outputSize;
    }

    proc init(outputSize: int) do
        this.init(real,outputSize);

    override proc forward(input: Tensor(eltType)): Tensor(eltType) {
        return input.adaptiveAvgPool2d(outputSize);
    }

    override proc attributes(): moduleAttributes {
      return new moduleAttributes(
            "AdaptiveAvgPool2D",
            moduleName,
            ("outputSize", outputSize));
    }
}

class Flatten : Module(?) {
    proc init(type eltType = real) do
        super.init(eltType);
    
    override proc forward(input: Tensor(eltType)): Tensor(eltType) do
        return input.flatten();

    override proc attributes(): moduleAttributes do
        return new moduleAttributes("Flatten",moduleName);
}

class ReLU : Module(?) {
    proc init(type eltType = real) do
        super.init(eltType);
    
    override proc forward(input: Tensor(eltType)): Tensor(eltType) do
        return input.relu();

    override proc attributes(): moduleAttributes do
        return new moduleAttributes("ReLU",moduleName);
}

class Softmax : Module(?) {

    proc init(type eltType = real) {
        super.init(eltType);
    }

    override proc forward(input: Tensor(eltType)): Tensor(eltType) do
        return input.softmax();

    override proc attributes(): moduleAttributes do
        return new moduleAttributes("SoftMax",moduleName);
}

// TODO: dropout is only valid for inference, since its a noop
class Dropout : Module(?) {
    proc init(type eltType = real,freq: real = 0.5) do
        super.init(eltType);

    override proc forward(input: Tensor(eltType)): Tensor(eltType) do
        return input; // dropout is not used for inference

    override proc attributes(): moduleAttributes {
        return new moduleAttributes(
            "Dropout",
            moduleName,
            ("frequency", 0.5));
    }
}



proc chain(m: borrowed Module(?), modNames: string...?n, input: Tensor(?eltType)) {
    var output = m.mod(modNames(0))(input);
    for param i in 1..<n {
        output = m.mod(modNames(i))(output);
    }
    return output;
}

class Net : Module(?) {
    proc init(type eltType = real) {
        super.init(eltType);
        init this;
        addModule("conv1",new Conv2D(eltType,3,32,3,stride=1));
        addModule("pool1",new MaxPool(eltType,2));
        addModule("conv2",new Conv2D(eltType,32,64,3,stride=1));
        addModule("pool2",new MaxPool(eltType,2));
        addModule("conv3",new Conv2D(eltType,64,128,3,stride=1));
        addModule("pool3",new MaxPool(eltType,2));
        addModule("conv4",new Conv2D(eltType,128,256,3,stride=1));
        addModule("pool4",new MaxPool(eltType,2));
        addModule("conv5",new Conv2D(eltType,256,512,3,stride=1));
        addModule("pool5",new MaxPool(eltType,2));
        addModule("conv6",new Conv2D(eltType,512,1024,3,stride=1));
        addModule("pool6",new MaxPool(eltType,2));
        // addModule("conv2",new Conv2D(32,64,3,stride=1));
        // addModule("")
    }

    override proc forward(input: Tensor(eltType)): Tensor(eltType) {
        return chain(this,
                    "conv1",
                    // "pool1",
                    "conv2",
                    // "pool2",
                    "conv3",
                    // "pool3",
                    "conv4",
                    // "pool4",
                    "conv5",
                    // "pool5",
                    "conv6",
                    // "pool6",
                    input);
        // var x1 = this.mod("conv1").forward(input);
        // var x2 = this.mod("conv2").forward(x1);
        // var x3 = this.mod("conv3").forward(x2);
        // return this.mod("conv4").forward(x3);
    }
}


if diag {
    use GpuDiagnostics;

    startGpuDiagnostics();
    startVerboseGpu();
}

proc main() {

    var flower = Tensor.load("data/flower.chdata");

    // var mp = new moduleChildren(real);

    var conv = new Conv2D(1,1,3,stride=1);
    var flat = new Flatten();
    var linear = new Linear(3,49);
    var relu = new ReLU();
    var softmax = new Softmax();

    // var model = new Squential(
    //     new Conv2D(1,1,3,stride=1),
    //     new Flatten(),
    //     new Linear(3,49),
    //     new ReLU(),
    //     new Softmax()
    // );

    var img = Tensor.arange(1,9,9);
    var fet = conv.forward(img);
    writeln(fet);

    var output = softmax(relu(linear(flat(conv(img)))));
    writeln(output);

    var t = Tensor.load("notebooks/mini_cnn_params.chdata");
    writeln(t);

    writeln(flower.tensorize(3).array.domain.shape);


    writeln("Instantiating network.");

    var net = new Net();
    // (net.subModules.childDict["conv1"].subModules.childDict["weights"] : borrowed Parameter(real)).data = Tensor.load("notebooks/mini_cnn_params.chdata");

    writeln("Feeding flower through network.");


    // var out_flower = net(flower);
    // writeln(out_flower.tensorize(3).array.domain.shape);

    // writeln(linear);

    import IO;
    import JSON;
    import ObjectSerialization;

    var objWriter = IO.stdout.withSerializer(ObjectSerialization.objectSerializer);
    var jsonWriter = IO.stdout.withSerializer(JSON.jsonSerializer);


    var a = ndarray.arange(15, real, (3,5));


    jsonWriter.writeln(a);
    objWriter.writeln(a);
    writeln(a);


    var b = staticTensor.arange(3,5);
    jsonWriter.writeln(b);
    objWriter.writeln(b);
    writeln(b);


    var c = Tensor.arange(3,5);
    jsonWriter.writeln(c);
    objWriter.writeln(c);
    writeln(c);

    var f = IO.open("myfile.txt", IO.ioMode.cw);
    var fw = f.writer();
    fw.writeln(c);
}
}
