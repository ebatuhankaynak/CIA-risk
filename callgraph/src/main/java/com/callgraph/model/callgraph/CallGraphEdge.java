package com.callgraph.model.callgraph;

import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

public class CallGraphEdge {
    private String startNodeSignature;
    private String endNodeSignature;

    public CallGraphEdge() {
    }

    public CallGraphEdge(String startNodeSignature, String endNodeSignature) {
        this.startNodeSignature = startNodeSignature;
        this.endNodeSignature = endNodeSignature;
    }

    public String getStartNodeSignature() {
        return startNodeSignature;
    }

    public void setStartNodeSignature(String startNodeSignature) {
        this.startNodeSignature = startNodeSignature;
    }

    public String getEndNodeSignature() {
        return endNodeSignature;
    }

    public void setEndNodeSignature(String endNodeSignature) {
        this.endNodeSignature = endNodeSignature;
    }

    @Override
    public String toString() {
        return "CallGraphEdge{" +
                "startNodeSignature='" + startNodeSignature + '\'' +
                ", endNodeSignature='" + endNodeSignature + '\'' +
                '}';
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof CallGraphEdge)) return false;
        CallGraphEdge that = (CallGraphEdge) o;
        return Objects.equals(startNodeSignature, that.startNodeSignature) && Objects.equals(endNodeSignature, that.endNodeSignature);
    }

    @Override
    public int hashCode() {
        return Objects.hash(startNodeSignature, endNodeSignature);
    }

    public Map<String, Object> toMap() {
        Map<String, Object> map = new HashMap<>();
        map.put("startNodeSignature", getStartNodeSignature());
        map.put("endNodeSignature", getEndNodeSignature());
        return map;
    }
}
