package com.callgraph.model.callgraph;

import java.util.ArrayList;
import java.util.HashMap;

public class CallGraph {
    private ArrayList<CallGraphNode> nodeList;
    private HashMap<String, CallGraphNode> nodeMap;
    private ArrayList<CallGraphEdge> edgeList;

    public CallGraph() {
    }

    public CallGraph(ArrayList<CallGraphNode> nodeList, HashMap<String, CallGraphNode> nodeMap, ArrayList<CallGraphEdge> edgeList) {
        this.nodeList = nodeList;
        this.nodeMap = nodeMap;
        this.edgeList = edgeList;
    }

    public CallGraph(HashMap<String, CallGraphNode> nodeMap, ArrayList<CallGraphEdge> edgeList) {
        this.nodeList = new ArrayList<>(nodeMap.values());
        this.nodeMap = nodeMap;
        this.edgeList = edgeList;
    }

    public ArrayList<CallGraphNode> getNodeList() {
        return nodeList;
    }

    public void setNodeList(ArrayList<CallGraphNode> nodeList) {
        this.nodeList = nodeList;
    }

    public HashMap<String, CallGraphNode> getNodeMap() {
        return nodeMap;
    }

    public void setNodeMap(HashMap<String, CallGraphNode> nodeMap) {
        this.nodeMap = nodeMap;
    }

    public ArrayList<CallGraphEdge> getEdgeList() {
        return edgeList;
    }

    public void setEdgeList(ArrayList<CallGraphEdge> edgeList) {
        this.edgeList = edgeList;
    }
}
