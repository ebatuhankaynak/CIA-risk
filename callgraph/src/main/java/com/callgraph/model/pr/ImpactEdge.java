package com.callgraph.model.pr;

import com.callgraph.model.callgraph.CallGraphEdge;

public class ImpactEdge extends CallGraphEdge {
    private Integer impactLevel;

    public ImpactEdge() {
    }

    public ImpactEdge(String startNodeSignature, String endNodeSignature, Integer impactLevel) {
        super(startNodeSignature, endNodeSignature);
        this.impactLevel = impactLevel;
    }

    public Integer getImpactLevel() {
        return impactLevel;
    }

    public void setImpactLevel(Integer impactLevel) {
        this.impactLevel = impactLevel;
    }
}
