package com.callgraph.model.general;

import java.util.Objects;

public class Tuple<A, B>{
    private A value0;
    private B value1;

    public Tuple() {
    }

    public Tuple(A value0, B value1) {
        this.value0 = value0;
        this.value1 = value1;
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

    @Override
    public String toString() {
        return "Tuple{" +
                "value0=" + value0 +
                ", value1=" + value1 +
                '}';
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof Tuple)) return false;
        Tuple<?, ?> tuple = (Tuple<?, ?>) o;
        return Objects.equals(value0, tuple.value0) && Objects.equals(value1, tuple.value1);
    }

    @Override
    public int hashCode() {
        return Objects.hash(value0, value1);
    }
}
