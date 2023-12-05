package com.callgraph.model.callgraph;

import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

public class CallGraphNode {
    private String className;
    private String functionName;
    private String packageName;
    private String functionSignature;
    private String filePath;
    private String simplifiedSignature;

    public CallGraphNode() {}

    public CallGraphNode(String className, String functionName, String packageName, String functionSignature, String filePath) {
        this.className = className;
        this.functionName = functionName;
        this.packageName = packageName;
        this.functionSignature = functionSignature;
        this.filePath = filePath;
        //this.simplifiedSignature = Utils.simplifySignature(functionSignature, true);
    }

    public String getClassName() {
        return className;
    }

    public void setClassName(String className) {
        this.className = className;
    }

    public String getFunctionName() {
        return functionName;
    }

    public void setFunctionName(String functionName) {
        this.functionName = functionName;
    }

    public String getPackageName() {
        return packageName;
    }

    public void setPackageName(String packageName) {
        this.packageName = packageName;
    }

    public String getFunctionSignature() {
        return functionSignature;
    }

    public void setFunctionSignature(String functionSignature) {
        this.functionSignature = functionSignature;
    }

    public String getFilePath() {
        return filePath;
    }

    public void setFilePath(String filePath) {
        this.filePath = filePath;
    }

    public String getSimplifiedSignature() {
        return simplifiedSignature;
    }

    public void setSimplifiedSignature(String simplifiedSignature) {
        this.simplifiedSignature = simplifiedSignature;
    }

    @Override
    public String toString() {
        return "CallGraphNode{" +
                "functionSignature='" + functionSignature + '\'' +
                '}';
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof CallGraphNode)) return false;
        CallGraphNode that = (CallGraphNode) o;
        return Objects.equals(functionSignature, that.functionSignature);
    }

    @Override
    public int hashCode() {
        return Objects.hash(functionSignature);
    }

    public Map<String, Object> toMap() {
        Map<String, Object> map = new HashMap<>();
        map.put("functionName", getFunctionName());
        map.put("className", getClassName());
        map.put("packageName", getPackageName());
        map.put("signature", getFunctionSignature());
        map.put("filePath", getFilePath());

        return map;
    }
}
