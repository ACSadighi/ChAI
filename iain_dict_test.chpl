use OrderedDict;



writeln("Hello, world!");

var d = new dict(
    ("one", 1),
    ("two", 2),
    ("three", 3)
);


for (k,v) in zip(d.keys(), d.values()) {
    writeln(k, " => ", v);
}



// increment the value for all
for (k,v) in zip(d.keys(),d.values()) {
    v += 1;
}


for (k,v) in zip(d.keys(), d.values()) {
    writeln(k, " => ", v);
}