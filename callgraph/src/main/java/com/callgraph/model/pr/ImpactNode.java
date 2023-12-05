package com.callgraph.model.pr;

import com.callgraph.model.callgraph.CallGraphNode;

public class ImpactNode extends CallGraphNode {
    private boolean isChanged;
    private boolean isDeleted;
    private boolean isAffected;
    private boolean isTerminal;

    public ImpactNode() {
    }

    public ImpactNode(String className, String functionName, String packageName, String functionSignature, String filePath, boolean isChanged, boolean isDeleted, boolean isAffected, boolean isTerminal) {
        super(className, functionName, packageName, functionSignature, filePath);
        this.isChanged = isChanged;
        this.isDeleted = isDeleted;
        this.isAffected = isAffected;
        this.isTerminal = isTerminal;
    }

    public boolean isChanged() {
        return isChanged;
    }

    public void setChanged(boolean changed) {
        isChanged = changed;
    }

    public boolean isDeleted() {
        return isDeleted;
    }

    public void setDeleted(boolean deleted) {
        isDeleted = deleted;
    }

    public boolean isAffected() {
        return isAffected;
    }

    public void setAffected(boolean affected) {
        isAffected = affected;
    }

    public boolean isTerminal() {
        return isTerminal;
    }

    public void setTerminal(boolean terminal) {
        isTerminal = terminal;
    }
}
