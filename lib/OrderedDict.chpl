



use List;
use Map;

record dict : serializable {
    type keyType;
    type valType;

    var table: map(keyType,valType);
    var order: list(keyType);

    proc init(in table: map(?keyType,?valType),in order: list(keyType)) {
        this.keyType = keyType;
        this.valType = valType;
        this.table = table;
        this.order = order;
    }

    proc init(in table: map(?keyType,?valType)) {
        var ks = new list(keyType);
        var tbl = new map(keyType,valType);
        for k in table.keys() {
            ks.pushBack(k);
            tbl.addOrReplace(k,table[k]);
        }
        this.init(table,ks);
    }

    proc init(type keyType, type valType) {
        this.keyType = keyType;
        this.valType = valType;
        this.table = new map(keyType,valType);
        this.order = new list(keyType);
    }

    proc init(entries: (?keyType,?valType) ...?n) {
        this.init(keyType,valType);
        for param i in 0..<n {
            this.insert(entries(i)[0],entries(i)[1]);
        }
    }

    proc size: int {
        const s = table.size;
        assert(s == order.size, "Table and order sizes out of sync: ", s, order.size);
        return s;
    }

    proc ref this(key: keyType) ref throws {
        if !table.contains(key) then
            throw new Error("Key not found: " + key:string);
        return table[key];
    }

    proc const this(key: keyType) const ref throws {
        if !table.contains(key) then
            throw new Error("Key not found: " + key:string);
        return table[key];
    }


    proc ref getKey(i: int) ref do
        return order[i];
    
    proc const ref getKey(i: int) const ref do
        return order[i];

    iter keys() do 
        for i in 0..<this.size do 
            yield this.getKey(i);

    proc ref getNVal(i: int) ref throws {
        return table[order[i]];
    }

    proc const ref getNVal(i: int) const ref throws {
        return table[order[i]];
    }

    iter ref values() ref do
        for i in 0..<this.size do 
            yield this.getNVal(i);

    iter const ref values() const ref do
        for i in 0..<this.size do 
            yield this.getNVal(i);

    // iter values() do
    //     for i in 0..<this.size do 
    //         yield this.getVal(i);

    proc ref this(k: keyType) ref throws {
        if !table.contains(k) then
            throw new Error("Key not found: " + k:string);
        return table[k];
    }

    // iter values() ref : valType do 
    //     for k in keys() do 
    //         yield table[k];

    


    // iter these() {
    //     for (k,v) in zip(this.keys(),this.values()) {
    //         yield (k,v);
    //     }
    // }


    // iter these() ref where !isClassType(valType) {
    //     for k in order {
    //         ref v = table[k];
    //         yield (k,v);
    //     }
    // }

    // iter these() const ref where !isClassType(valType) do

    //     for k in order {
    //         const ref v = table[k];
    //         yield (k,v);
    //     }

    // iter these() where !isClassType(valType) do
    //     for k in order {
    //         yield (k,table[k]);
    //     }
    
    // iter these() where isSharedClassType(valType) {
    //     for k in order {
    //         yield (k,table[k]);
    //     }
    // }

    // // iter these() where isClassType(valType) {
    // //     compilerError(valType:string);
    // //     for k in order {
    // //         yield (k,table[k]);
    // //     }
    // // }

    // iter these() where isClassType(valType) && !isSharedClassType(valType) {
    //     compilerError(valType:string);
    //     for k in order {
    //         yield (k,table[k]);
    //     }
    // }
    
    // // iter these() ref {
    // //     for k in order {
    // //         ref v = table[k];
    // //         yield (k,v);
    // //     }
    // // }


    // // iter these() const ref {
    // //     for k in order {
    // //         const ref v = table[k];
    // //         yield (k,v);
    // //     }
    // // }


    proc ref insert(in key: keyType, in value: valType) {
        if !order.contains(key) then
            order.pushBack(key);
        table.addOrReplace(key,value);
    }

    proc createWith(in key: keyType, in value: valType): dict(keyType,valType) {
        var newDict = this;
        newDict.insert(key,value);
        return newDict;
    }

    proc ref remove(in key: keyType) {
        table.remove(key);
        order.remove(key);
    }

    proc createWithout(in key: keyType): dict(keyType,valType) {
        var ord: list(keyType);
        var tbl: map(keyType,valType);
        for k in this.keys() {
            if k != key then {
                ord.pushBack(k);
                tbl.addOrReplace(k,this[k]);
            }
        }
        return new dict(tbl,ord);
    }

}