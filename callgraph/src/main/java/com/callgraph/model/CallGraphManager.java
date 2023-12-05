package com.callgraph.model;

import com.callgraph.model.callgraph.CallGraph;
import com.callgraph.model.callgraph.CallGraphEdge;
import com.callgraph.model.callgraph.CallGraphNode;
import org.neo4j.driver.*;
import org.neo4j.driver.exceptions.Neo4jException;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.stream.Collectors;

import static org.neo4j.driver.Values.parameters;

public class CallGraphManager implements AutoCloseable {
    private static final Logger LOGGER = Logger.getLogger(CallGraphManager.class.getName());
    private final Driver driver;

    public CallGraphManager() {
        // The driver is a long living object and should be opened during the start of your application
        String uri = "";
        String user = "";
        String password = "";
        driver = GraphDatabase.driver(uri, AuthTokens.basic(user, password), Config.defaultConfig());
    }

    @Override
    public void close() {
        // The driver object should be closed before the application ends.
        driver.close();
    }

    public void createCallGraph(CallGraph callGraph, String branchName, String projectIdentifier) {
        ArrayList<CallGraphNode> nodes = callGraph.getNodeList();
        ArrayList<CallGraphEdge> edges = callGraph.getEdgeList();

        List<Map<String, Object>> mapOfNodes = nodes.stream().map(CallGraphNode::toMap).collect(Collectors.toList());
        List<Map<String, Object>> mapOfEdges = edges.stream().map(CallGraphEdge::toMap).collect(Collectors.toList());

        Query nodeQuery = new Query("UNWIND $nodes AS nodes \n"
                                  + "CREATE (f:Function) SET f = nodes, f.branchName = $branchName, \n"
                                  + "f.projectIdentifier = $projectIdentifier ",
                                  parameters("nodes", mapOfNodes, "branchName", branchName, "projectIdentifier", projectIdentifier));

        try (Session session = driver.session(SessionConfig.forDatabase("neo4j"))) {
            session.writeTransaction(tx -> tx.run(nodeQuery));
        } catch (Neo4jException ex) {
            LOGGER.log(Level.SEVERE, nodeQuery + " raised an exception", ex);
            throw ex;
        }

        Query edgeQuery = new Query("UNWIND $edges AS edges \n"
                                    + "MATCH (a:Function { signature: edges.startNodeSignature, branchName: $branchName, projectIdentifier: $projectIdentifier }), (b:Function { signature: edges.endNodeSignature, branchName: $branchName, projectIdentifier: $projectIdentifier }) \n"
                                    + "CREATE (a)-[:CALLS]->(b) \n",
                                      parameters("edges", mapOfEdges, "branchName", branchName, "projectIdentifier", projectIdentifier));

        try (Session session = driver.session(SessionConfig.forDatabase("neo4j"))) {
            session.writeTransaction(tx -> tx.run(edgeQuery));
        } catch (Neo4jException ex) {
            LOGGER.log(Level.SEVERE, edgeQuery + " raised an exception", ex);
            throw ex;
        }
    }

    public void updateCallGraph(List<CallGraphNode> removedNodes,
                                List<CallGraphNode> addedNodes,
                                List<CallGraphEdge> removedEdges,
                                List<CallGraphEdge> addedEdges,
                                String projectIdentifier,
                                String destinationBranchName) {

        if (!removedNodes.isEmpty()) {
            List<String> removedNodeSig = removedNodes.stream().map(CallGraphNode::getFunctionSignature).collect(Collectors.toList());

            Query query = new Query("MATCH (f:Function) WHERE f.projectIdentifier = $projectIdentifier AND f.branchName = $branchName AND f.signature IN $removedNodeSig \n"
                                    + "WITH COLLECT(f) AS nodes \n"
                                    + "FOREACH (node IN nodes | DETACH DELETE node)",
                                    parameters("projectIdentifier", projectIdentifier, "branchName", destinationBranchName, "removedNodeSig", removedNodeSig));

            try (Session session = driver.session(SessionConfig.forDatabase("neo4j"))) {
                session.writeTransaction(tx -> tx.run(query));
            } catch (Neo4jException ex) {
                LOGGER.log(Level.SEVERE, query + " raised an exception", ex);
                throw ex;
            }
        }

        if (!addedNodes.isEmpty()) {
            List<Map<String, Object>> mapOfNodes = addedNodes.stream().map(CallGraphNode::toMap).collect(Collectors.toList());

            Query query = new Query("UNWIND $nodes AS nodes \n"
                                    + "CREATE (f:Function) SET f = nodes, f.branchName = $branchName, f.projectIdentifier = $projectIdentifier",
                                    parameters("nodes", mapOfNodes, "branchName", destinationBranchName, "projectIdentifier", projectIdentifier));

            try (Session session = driver.session(SessionConfig.forDatabase("neo4j"))) {
                session.writeTransaction(tx -> tx.run(query));
            } catch (Neo4jException ex) {
                LOGGER.log(Level.SEVERE, query + " raised an exception", ex);
                throw ex;
            }
        }

        if (!removedEdges.isEmpty()) {
            List<Map<String, Object>> mapOfEdges = removedEdges.stream().map(CallGraphEdge::toMap).collect(Collectors.toList());

            Query query = new Query("UNWIND $edges AS edges \n"
                                    + "MATCH (startNode:Function)-[r:CALLS]->(endNode:Function) \n"
                                    + "WHERE startNode.signature = edges.startNodeSignature AND endNode.signature = edges.endNodeSignature \n"
                                    + "AND startNode.branchName = $branchName AND startNode.projectIdentifier = $projectIdentifier \n"
                                    + "AND endNode.branchName = $branchName AND endNode.projectIdentifier = $projectIdentifier \n"
                                    + "DELETE r", parameters("edges", mapOfEdges, "branchName", destinationBranchName, "projectIdentifier", projectIdentifier));

            try (Session session = driver.session(SessionConfig.forDatabase("neo4j"))) {
                session.writeTransaction(tx -> tx.run(query));
            } catch (Neo4jException ex) {
                LOGGER.log(Level.SEVERE, query + " raised an exception", ex);
                throw ex;
            }
        }

        if (!addedEdges.isEmpty()) {
            List<Map<String, Object>> mapOfEdges = addedEdges.stream().map(CallGraphEdge::toMap).collect(Collectors.toList());

            Query query = new Query("UNWIND $edges AS edges \n"
                                    + "MATCH (a:Function { signature: edges.startNodeSignature, branchName: $branchName, projectIdentifier: $projectIdentifier }), (b:Function { signature: edges.endNodeSignature, branchName: $branchName, projectIdentifier: $projectIdentifier }) \n"
                                    + "CREATE (a)-[:CALLS]->(b) \n",
                                    parameters("edges", mapOfEdges, "branchName", destinationBranchName, "projectIdentifier", projectIdentifier));

            try (Session session = driver.session(SessionConfig.forDatabase("neo4j"))) {
                session.writeTransaction(tx -> tx.run(query));
            } catch (Neo4jException ex) {
                LOGGER.log(Level.SEVERE, query + " raised an exception", ex);
                throw ex;
            }
        }
    }
}