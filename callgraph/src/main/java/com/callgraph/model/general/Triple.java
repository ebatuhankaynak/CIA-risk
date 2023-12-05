package com.callgraph.model.general;

import java.util.Objects;

public class Triple<A, B, C> {
    private A value0;
    private B value1;
    private C value2;

    public Triple() {
    }

    public Triple(A value0, B value1, C value2) {
        this.value0 = value0;
        this.value1 = value1;
        this.value2 = value2;
    }

    public A getValue0() {
        return value0;
    }

    public void setValue0(A value0) {
        this.value0 = value0;
    }

    public B getValue1() {
        return value1;
    }

    public void setValue1(B value1) {
        this.value1 = value1;
    }

    public C getValue2() {
        return value2;
    }

    public void setValue2(C value2) {
        this.value2 = value2;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof Triple)) return false;
        Triple<?, ?, ?> triple = (Triple<?, ?, ?>) o;
        return Objects.equals(value0, triple.value0) && Objects.equals(value1, triple.value1) && Objects.equals(value2, triple.value2);
    }

    @Override
    public int hashCode() {
        return Objects.hash(value0, value1, value2);
    }
}
